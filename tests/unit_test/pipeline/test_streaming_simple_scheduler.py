# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
import threading
import time
from collections import Counter

import pytest
import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from tests.unit_test.scheduling.test_streaming_vocoder import _FakeStreamingVocoder


def _payload(request_id: str, *, stream: bool = False) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=[], params={"stream": stream}),
        data={"request_id": request_id},
    )


class _TestStreamingScheduler(StreamingSimpleScheduler):
    def __init__(self, *, max_batch_size: int = 4, max_batch_wait_ms: int = 0):
        self.single_calls: list[str] = []
        self.batch_calls: list[list[str]] = []
        self.stream_state: set[str] = set()
        super().__init__(
            self._compute,
            batch_compute_fn=self._compute_batch,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return bool(payload.request.params.get("stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        del payload
        self.stream_state.add(request_id)

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        self.stream_state.add(request_id)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data={"chunk": item.data},
                metadata={"modality": "test"},
            )
        ]

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        return [
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data={"done": request_id},
            )
        ]

    def clear_stream_state(self, request_id: str) -> None:
        self.stream_state.discard(request_id)

    def _compute(self, payload: StagePayload) -> StagePayload:
        self.single_calls.append(payload.request_id)
        payload.data = {"single": payload.request_id}
        return payload

    def _compute_batch(self, payloads: list[StagePayload]) -> list[StagePayload]:
        self.batch_calls.append([payload.request_id for payload in payloads])
        for payload in payloads:
            payload.data = {"batch": payload.request_id}
        return payloads


def _drain_results(scheduler: StreamingSimpleScheduler) -> list[OutgoingMessage]:
    messages: list[OutgoingMessage] = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def _run_serving_loop(scheduler: StreamingSimpleScheduler, *, limit: int = 8) -> None:
    """Dispatch queued messages the way the serving loop does.

    Bounded so a message the scheduler keeps re-queueing fails the calling
    assertion instead of hanging the test session.
    """
    for _ in range(limit):
        if not scheduler._pending_messages and scheduler.inbox.empty():
            return
        scheduler._handle_message(scheduler._next_message(), None)
    raise AssertionError("scheduler did not settle; a queued message livelocked")


def _serve(
    scheduler: StreamingSimpleScheduler,
    messages: list[IncomingMessage],
    *,
    output_count: int,
) -> list[OutgoingMessage]:
    """Run the real serving loop in a thread and collect ``output_count``
    messages from the outbox; ``queue.Empty`` means the loop stopped early.

    Inlined from ``tests.unit_test.pipeline.helpers.run_scheduler`` for the same
    reason as ``test_simple_scheduler_concurrent.py``: that module drags in the
    config/placement import graph.
    """
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for message in messages:
            scheduler.inbox.put(message)
        return [scheduler.outbox.get(timeout=2.0) for _ in range(output_count)]
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)


def _deferred_stream_done_without_its_chunk(
    scheduler: StreamingSimpleScheduler, request_id: str
) -> None:
    """Leave ``request_id`` with a deferred done whose wait bound has passed.

    The deferral is created the way the collector does it: one of the request's
    own chunks is parked and its done handed to the scheduler. The parked chunk
    is then taken out of the queue (a later serve step, or the vocoder's own
    drain, consumes it), so nothing is left for the request but the deferred
    done.
    """
    scheduler._on_streaming_new_request(request_id, _payload(request_id, stream=True))
    scheduler._pending_messages.append(_chunk(request_id, "older"))
    scheduler._on_done(request_id)
    assert request_id in scheduler._deferred_done_deadlines
    scheduler._pending_messages.clear()
    scheduler._deferred_done_deadlines[request_id] = time.monotonic() - 1.0


class _FailingDoneStreamingScheduler(_TestStreamingScheduler):
    """``on_stream_done`` fails for ``failing_done_ids`` (a failing final decode)."""

    def __init__(
        self,
        *,
        failing_done_ids: set[str],
        batched: bool = False,
        **kwargs: int,
    ) -> None:
        self.failing_done_ids = set(failing_done_ids)
        self._can_batch_stream_chunks = batched
        super().__init__(**kwargs)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        if request_id in self.failing_done_ids:
            raise RuntimeError(f"final decode failed for {request_id!r}")
        return super().on_stream_done(request_id)


def _vocoder_item(chunk_id: int, frames: list[int]) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=torch.tensor(frames, dtype=torch.long),
        from_stage="src",
        metadata={"modality": "audio_codes", "stream": True},
    )


def _waveform_values(message: OutgoingMessage) -> list[float]:
    data = message.data
    return torch.frombuffer(
        bytearray(data["audio_waveform"]), dtype=torch.float32
    ).tolist()


class _OrderedFakeStreamingVocoder(_FakeStreamingVocoder):
    """The shared base vocoder with per-request chunk-id ordering enabled."""

    _enforce_chunk_id_order = True


def test_idle_resume_of_a_failing_deferred_done_keeps_the_loop_alive() -> None:
    # start() drains deferred dones on idle outside its try/except, so a failing
    # final decode there must be isolated: error for that request, that request
    # aborted, its deferral bookkeeping cleared, and the stage still serving.
    scheduler = _FailingDoneStreamingScheduler(failing_done_ids={"req-a"})
    _deferred_stream_done_without_its_chunk(scheduler, "req-a")

    failures: list[BaseException] = []
    try:
        out = _serve(scheduler, [], output_count=1)
    except queue.Empty as exc:
        failures.append(exc)

    assert (
        failures == []
    ), "the idle serving loop died instead of emitting an error for req-a"
    assert [(message.type, message.request_id) for message in out] == [
        ("error", "req-a")
    ]
    assert scheduler._is_aborted("req-a")
    assert "req-a" not in scheduler._deferred_done_deadlines
    assert "req-a" not in scheduler._pending_done

    # The failure is consumed with the request: a later resume must neither
    # re-raise nor emit a second error.
    scheduler._resume_deferred_stream_done()
    assert _drain_results(scheduler) == []


@pytest.mark.parametrize("batched", [False, True])
def test_deferred_done_failure_is_not_attributed_to_an_innocent_request(
    batched: bool,
) -> None:
    # A chunk delivery resumes deferred dones. When one of those dones fails, the
    # error belongs to the failing request: the request whose chunk triggered the
    # resume must keep streaming.
    scheduler = _FailingDoneStreamingScheduler(
        failing_done_ids={"req-a"}, batched=batched
    )
    _deferred_stream_done_without_its_chunk(scheduler, "req-a")
    scheduler._on_streaming_new_request("req-b", _payload("req-b", stream=True))

    out = _serve(scheduler, [_chunk("req-b", "chunk")], output_count=2)

    assert [(message.type, message.request_id) for message in out] == [
        ("stream", "req-b"),
        ("error", "req-a"),
    ]
    assert scheduler._is_aborted("req-a")
    assert not scheduler._is_aborted("req-b")
    assert "req-b" in scheduler.stream_state
    assert "req-a" not in scheduler._deferred_done_deadlines


def test_base_vocoder_defers_a_done_while_it_holds_a_chunk_out_of_order() -> None:
    # `_deferred_chunks` lives in StreamingVocoderBase, so the "this request still
    # holds a chunk" predicate must live there too: a check kept in one model
    # subclass leaves every other chunk-id-ordered vocoder completing early and
    # dropping the held chunk.
    scheduler = _OrderedFakeStreamingVocoder()
    scheduler._on_streaming_new_request("req-a", _payload("req-a", stream=True))
    scheduler._on_chunk("req-a", _vocoder_item(1, [3, 4]))
    assert scheduler._deferred_chunks.get("req-a")

    scheduler._on_done("req-a")

    # The held chunk is still unprocessed, so the done waits instead of
    # finalizing without it and clearing the request state.
    assert _drain_results(scheduler) == []
    assert "req-a" in scheduler._deferred_done_deadlines
    assert "req-a" in scheduler._stream_states

    # The missing head lands: the held chunk is ingested after it and the
    # deferred done completes exactly once with both chunks, in order.
    scheduler._on_chunk("req-a", _vocoder_item(0, [1, 2]))
    out = _drain_results(scheduler)
    assert [message.type for message in out] == ["stream", "result"]
    assert out[-1].request_id == "req-a"
    assert out[-1].data.data["frames"] == 2
    assert _waveform_values(out[0]) == [1.0, 2.0, 3.0, 4.0]
    assert "req-a" not in scheduler._deferred_done_deadlines
    assert "req-a" not in scheduler._pending_done
    assert not scheduler._deferred_chunks.get("req-a")


def test_streaming_simple_scheduler_batches_non_streaming_requests() -> None:
    scheduler = _TestStreamingScheduler(max_batch_size=3)
    first = IncomingMessage("a", "new_request", _payload("a"))
    scheduler.inbox.put(IncomingMessage("b", "new_request", _payload("b")))
    scheduler.inbox.put(IncomingMessage("c", "new_request", _payload("c")))

    batch = scheduler._collect_new_request_batch(first)
    scheduler._handle_new_request_batch(batch)

    assert scheduler.batch_calls == [["a", "b", "c"]]
    assert [msg.request_id for msg in _drain_results(scheduler)] == ["a", "b", "c"]


def test_non_streaming_batch_skips_done_before_later_payloads() -> None:
    scheduler = _TestStreamingScheduler(max_batch_size=3)
    first = IncomingMessage("a", "new_request", _payload("a"))
    for msg in (
        IncomingMessage("b", "stream_done"),
        IncomingMessage("b", "new_request", _payload("b")),
        IncomingMessage("c", "stream_done"),
        IncomingMessage("c", "new_request", _payload("c")),
    ):
        scheduler.inbox.put(msg)

    batch = scheduler._collect_new_request_batch(first)
    scheduler._handle_new_request_batch(batch)
    while scheduler._pending_messages:
        scheduler._handle_message(scheduler._next_message(), None)

    assert scheduler.batch_calls == [["a", "b", "c"]]
    assert [msg.request_id for msg in _drain_results(scheduler)] == ["a", "b", "c"]
    assert not scheduler._pending_messages
    assert not scheduler._pending_done


def test_non_streaming_batch_cost_limit_bounds_the_inbox_batch() -> None:
    # Ordering is request-local, not a global FIFO: collection reads the inbox
    # eagerly (that is what keeps first-audio latency low) and leaves the parked
    # deque to `_next_message`.
    scheduler = _TestStreamingScheduler(max_batch_size=4)
    scheduler._request_cost_fn = lambda payload: payload.data["cost"]
    scheduler._max_batch_cost = 3
    requests = []
    for rid, cost in (("a", 1), ("b", 2), ("c", 3), ("d", 2), ("e", 1)):
        payload = _payload(rid)
        payload.data["cost"] = cost
        requests.append(IncomingMessage(rid, "new_request", payload))
    parked = [IncomingMessage("b", "stream_done"), requests[1], requests[2]]
    scheduler._pending_messages.extend(parked)
    scheduler.inbox.put(requests[3])
    scheduler.inbox.put(requests[4])

    batch = scheduler._collect_new_request_batch(requests[0])

    # a (1) + d (2) fill the cost budget; e (1) would exceed it.
    assert batch == [requests[0], requests[3]]
    # Nothing is lost: the parked messages stay, the rejected message is kept.
    assert Counter(
        (msg.request_id, msg.type) for msg in scheduler._pending_messages
    ) == (
        Counter(
            [
                ("b", "stream_done"),
                ("b", "new_request"),
                ("c", "new_request"),
                ("e", "new_request"),
            ]
        )
    )
    assert scheduler.inbox.empty()

    scheduler._handle_new_request_batch(batch)
    _run_serving_loop(scheduler, limit=8)

    assert scheduler.batch_calls == [["a", "d"]]
    # Deferral keeps arrival order: e was rejected after b and c were already
    # parked, so it is served behind them (front-inserting a deferral would make
    # the deque LIFO and postpone the oldest work indefinitely).
    assert scheduler.single_calls == ["b", "c", "e"]
    assert sorted(msg.request_id for msg in _drain_results(scheduler)) == [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]
    assert not scheduler._pending_done


def test_non_streaming_batch_reads_the_inbox_past_a_parked_done_marker() -> None:
    scheduler = _TestStreamingScheduler(max_batch_size=3)
    scheduler._on_streaming_new_request("stream", _payload("stream", stream=True))
    first = IncomingMessage("a", "new_request", _payload("a"))
    parked_done = IncomingMessage("stream", "stream_done")
    parked_payload = IncomingMessage("b", "new_request", _payload("b"))
    newer = IncomingMessage("c", "new_request", _payload("c"))
    scheduler._pending_messages.extend([parked_done, parked_payload])
    scheduler.inbox.put(newer)

    assert scheduler._collect_new_request_batch(first) == [first, newer]
    assert list(scheduler._pending_messages) == [parked_done, parked_payload]
    assert scheduler.inbox.empty()


def test_stream_done_defers_while_a_chunk_of_the_same_request_is_parked() -> None:
    scheduler = _TestStreamingScheduler()
    scheduler._on_streaming_new_request("req", _payload("req", stream=True))
    parked = _chunk("req", "older")
    scheduler._pending_messages.append(parked)

    scheduler._on_done("req")

    # The request still has an unprocessed chunk of its own parked ahead of the
    # completion, so the done waits instead of clearing the request state (which
    # would drop that chunk).
    assert _drain_results(scheduler) == []
    assert scheduler.stream_state == {"req"}
    assert any(message is parked for message in scheduler._pending_messages)

    # The parked chunk is served first; the deferred done is re-dispatched after it.
    _run_serving_loop(scheduler)

    out = _drain_results(scheduler)
    assert [message.type for message in out] == ["stream", "result"]
    assert scheduler.stream_state == set()
    assert not scheduler._pending_done


def test_streaming_simple_scheduler_keeps_streaming_request_out_of_batch() -> None:
    scheduler = _TestStreamingScheduler(max_batch_size=3)
    first = IncomingMessage("a", "new_request", _payload("a"))
    scheduler.inbox.put(
        IncomingMessage("stream", "new_request", _payload("stream", stream=True))
    )
    scheduler.inbox.put(IncomingMessage("b", "new_request", _payload("b")))

    batch = scheduler._collect_new_request_batch(first)

    assert [msg.request_id for msg in batch] == ["a"]
    assert scheduler._next_message().request_id == "stream"
    assert scheduler._next_message().request_id == "b"


def test_streaming_simple_scheduler_done_before_payload_finalizes_later() -> None:
    scheduler = _TestStreamingScheduler()

    scheduler._on_done("req")
    scheduler._on_streaming_new_request("req", _payload("req", stream=True))

    out = scheduler.outbox.get_nowait()
    assert out.type == "result"
    assert out.data == {"done": "req"}
    assert "req" not in scheduler._pending_done
    assert "req" not in scheduler.stream_state


def test_streaming_simple_scheduler_ignores_late_non_streaming_done() -> None:
    scheduler = _TestStreamingScheduler()

    scheduler._handle_new_request_batch(
        [IncomingMessage("req", "new_request", _payload("req", stream=False))]
    )
    scheduler._on_done("req")

    assert scheduler.outbox.get_nowait().type == "result"
    assert "req" not in scheduler._pending_done


def test_streaming_simple_scheduler_abort_clears_all_stream_state() -> None:
    scheduler = _TestStreamingScheduler()
    scheduler._stream_payloads["req"] = _payload("req", stream=True)
    scheduler._pending_done.add("req")
    scheduler.stream_state.add("req")

    scheduler.abort("req")

    assert "req" not in scheduler._stream_payloads
    assert "req" not in scheduler._pending_done
    assert "req" not in scheduler.stream_state
    assert "req" in scheduler._aborted_request_ids


def test_streaming_simple_scheduler_keeps_queued_control_message_out_of_batch() -> None:
    scheduler = _TestStreamingScheduler(max_batch_size=3)
    first = IncomingMessage("a", "new_request", _payload("a"))
    chunk = StreamItem(chunk_id=0, data="x", from_stage="source")
    scheduler.inbox.put(IncomingMessage("stream", "stream_chunk", chunk))
    scheduler.inbox.put(IncomingMessage("b", "new_request", _payload("b")))

    batch = scheduler._collect_new_request_batch(first)

    assert [msg.request_id for msg in batch] == ["a"]
    next_msg = scheduler._next_message()
    assert next_msg.request_id == "stream"
    assert next_msg.type == "stream_chunk"
    assert scheduler._next_message().request_id == "b"


def _chunk(request_id: str, value: str) -> IncomingMessage:
    return IncomingMessage(
        request_id, "stream_chunk", StreamItem(chunk_id=0, data=value, from_stage="src")
    )


def _raw_chunk(request_id: str, value: object) -> IncomingMessage:
    return IncomingMessage(request_id, "stream_chunk", value)


class _BatchStreamingScheduler(_TestStreamingScheduler):
    _can_batch_stream_chunks = True

    def __init__(self, **kw: int) -> None:
        self.pump_batches: list[list[str]] = []
        super().__init__(**kw)

    def on_stream_chunk_batch(self, items):
        self.pump_batches.append([rid for rid, _ in items])
        for request_id, item in items:
            if self._is_aborted(request_id):
                continue
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data={"chunk": item.data},
                    metadata={"modality": "test"},
                )
            )


class _DefaultBatchScheduler(_TestStreamingScheduler):
    _can_batch_stream_chunks = True


class _DistinctBatchStreamingScheduler(_BatchStreamingScheduler):
    _stream_chunk_batch_distinct_requests = True


def test_stream_chunk_batch_opt_out_dispatches_one_at_a_time() -> None:
    scheduler = _TestStreamingScheduler(max_batch_size=4)
    scheduler.inbox.put(_chunk("b", "y"))
    scheduler._handle_message(_chunk("a", "x"), None)
    assert [m.request_id for m in _drain_results(scheduler)] == ["a"]
    assert scheduler.inbox.get_nowait().request_id == "b"


def test_stream_chunk_batch_coalesces_queued_chunks_into_one_pump() -> None:
    scheduler = _BatchStreamingScheduler(max_batch_size=4)
    scheduler.inbox.put(_chunk("b", "y"))
    scheduler.inbox.put(_chunk("c", "z"))
    scheduler._handle_message(_chunk("a", "x"), None)
    assert scheduler.pump_batches == [["a", "b", "c"]]
    assert [m.data["chunk"] for m in _drain_results(scheduler)] == ["x", "y", "z"]


def test_stream_chunk_batch_coalesces_queued_inbox_chunks_only() -> None:
    # Eager inbox reads: peers queued behind the first chunk coalesce into one
    # pump, while parked messages stay for the serving loop. Reading the parked
    # deque first is what serialised the pipeline.
    scheduler = _BatchStreamingScheduler(max_batch_size=3)
    chunks = [_chunk(rid, rid) for rid in ("a", "b", "c", "d", "e")]
    parked = chunks[1:3]
    scheduler._pending_messages.extend(parked)
    for msg in chunks[3:]:
        scheduler.inbox.put(msg)

    scheduler._handle_message(chunks[0], None)

    assert scheduler.pump_batches == [["a", "d", "e"]]
    assert [m.data["chunk"] for m in _drain_results(scheduler)] == ["a", "d", "e"]
    assert list(scheduler._pending_messages) == parked
    assert scheduler.inbox.empty()


def test_stream_chunk_batch_can_stop_before_duplicate_request() -> None:
    scheduler = _DistinctBatchStreamingScheduler(max_batch_size=4)
    scheduler.inbox.put(_chunk("b", "y"))
    scheduler.inbox.put(_chunk("a", "second"))
    scheduler.inbox.put(_chunk("c", "z"))

    scheduler._handle_message(_chunk("a", "first"), None)

    assert scheduler.pump_batches == [["a", "b"]]
    assert scheduler._next_message().data.data == "second"
    assert scheduler._next_message().request_id == "c"


def test_stream_chunk_batch_stops_at_non_chunk_and_pushes_back() -> None:
    scheduler = _BatchStreamingScheduler(max_batch_size=4)
    scheduler.inbox.put(_chunk("b", "y"))
    scheduler.inbox.put(IncomingMessage("c", "new_request", _payload("c")))
    scheduler.inbox.put(_chunk("d", "w"))
    scheduler._handle_message(_chunk("a", "x"), None)
    assert scheduler.pump_batches == [["a", "b"]]
    assert scheduler._next_message().request_id == "c"
    assert scheduler._next_message().request_id == "d"


def test_stream_chunk_batch_skips_aborted_requests() -> None:
    scheduler = _BatchStreamingScheduler(max_batch_size=4)
    scheduler.abort("b")
    scheduler.inbox.put(_chunk("b", "y"))
    scheduler.inbox.put(_chunk("c", "z"))
    scheduler._handle_message(_chunk("a", "x"), None)
    assert scheduler.pump_batches == [["a", "c"]]
    assert [m.request_id for m in _drain_results(scheduler)] == ["a", "c"]


def test_stream_chunk_batch_respects_cap() -> None:
    scheduler = _BatchStreamingScheduler(max_batch_size=2)
    for rid in ("b", "c", "d"):
        scheduler.inbox.put(_chunk(rid, rid))
    scheduler._handle_message(_chunk("a", "x"), None)
    assert scheduler.pump_batches == [["a", "b"]]
    assert scheduler._next_message().request_id == "c"


def test_deferred_messages_keep_arrival_order() -> None:
    # Deferral used to front-insert, so the newest deferred message was served
    # first and the oldest could be postponed indefinitely. Each collector pass
    # below parks the marker behind its chunk; the queue must pop them in
    # arrival order.
    scheduler = _TestStreamingScheduler(max_batch_size=3)
    for rid in ("a", "b", "c"):
        first = _chunk(rid, rid)
        scheduler.inbox.put(IncomingMessage(rid, "stream_done"))
        assert scheduler._collect_stream_chunk_batch(first) == [first]

    assert [msg.request_id for msg in scheduler._pending_messages] == ["a", "b", "c"]
    assert [scheduler._next_message().request_id for _ in range(3)] == [
        "a",
        "b",
        "c",
    ]


def test_deferred_messages_use_the_arrival_stamp_when_present() -> None:
    # Inbox instrumentation stamps arrivals, so a deferral must land by stamp:
    # re-deferring an older message must not place it behind newer ones.
    scheduler = _TestStreamingScheduler(max_batch_size=3)
    for rid, arrived_at in (("c", 30.0), ("a", 10.0), ("b", 20.0)):
        first = _chunk(rid, rid)
        marker = IncomingMessage(rid, "stream_done")
        object.__setattr__(marker, "arrived_at", arrived_at)
        scheduler.inbox.put(marker)
        assert scheduler._collect_stream_chunk_batch(first) == [first]

    assert [msg.request_id for msg in scheduler._pending_messages] == ["a", "b", "c"]


def test_stream_chunk_batch_default_hook_emits_per_chunk_in_order() -> None:
    scheduler = _DefaultBatchScheduler(max_batch_size=4)
    scheduler.inbox.put(_chunk("b", "y"))
    scheduler._handle_message(_chunk("a", "x"), None)
    assert [m.data["chunk"] for m in _drain_results(scheduler)] == ["x", "y"]


class _RaisingDefaultBatchScheduler(_TestStreamingScheduler):
    _can_batch_stream_chunks = True

    def on_stream_chunk(self, request_id, item):
        if request_id == "bad":
            raise ValueError("boom")
        return super().on_stream_chunk(request_id, item)


def test_stream_chunk_batch_default_hook_isolates_failing_item() -> None:
    scheduler = _RaisingDefaultBatchScheduler(max_batch_size=4)
    scheduler.inbox.put(_chunk("bad", "y"))
    scheduler.inbox.put(_chunk("c", "z"))
    scheduler._handle_message(_chunk("a", "x"), None)
    out = _drain_results(scheduler)
    assert [m.request_id for m in out if m.type == "stream"] == ["a", "c"]
    assert any(m.request_id == "bad" and m.type == "error" for m in out)
    assert scheduler._is_aborted("bad")


def test_stream_chunk_batch_validates_items_before_hook() -> None:
    scheduler = _BatchStreamingScheduler(max_batch_size=4)
    scheduler.inbox.put(_raw_chunk("bad", "not-a-stream-item"))
    scheduler.inbox.put(_chunk("c", "z"))

    scheduler._handle_message(_chunk("a", "x"), None)

    out = _drain_results(scheduler)
    assert scheduler.pump_batches == [["a", "c"]]
    assert [m.request_id for m in out if m.type == "stream"] == ["a", "c"]
    assert any(m.request_id == "bad" and m.type == "error" for m in out)
    assert scheduler._is_aborted("bad")


def test_stream_chunk_batch_filters_request_aborted_during_validation() -> None:
    scheduler = _DefaultBatchScheduler(max_batch_size=4)
    scheduler.inbox.put(_raw_chunk("bad", "not-a-stream-item"))
    scheduler.inbox.put(_chunk("c", "z"))

    scheduler._handle_message(_chunk("bad", "x"), None)

    out = _drain_results(scheduler)
    assert [m.request_id for m in out if m.type == "stream"] == ["c"]
    assert any(m.request_id == "bad" and m.type == "error" for m in out)
    assert scheduler._is_aborted("bad")
    assert "bad" not in scheduler.stream_state
