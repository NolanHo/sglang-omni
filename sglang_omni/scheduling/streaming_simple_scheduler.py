# SPDX-License-Identifier: Apache-2.0
"""Shared scheduler for simple stages that process streaming chunks.

This keeps SimpleScheduler's inbox/outbox contract, but adds the request
lifecycle needed by stream-processing stages such as vocoders:

- ``new_request`` for both streaming setup and non-streaming compute
- ``stream_chunk`` carrying a :class:`StreamItem`
- ``stream_done`` that may arrive before the terminal payload
- non-streaming single and batched compute
"""

from __future__ import annotations

import asyncio
import collections
import logging
import queue as _queue_mod
import threading
import time
from typing import Any, Callable

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_ABORTED_REQUEST_ID_LIMIT = 10000
_ABORTED_REQUEST_ID_RETAINED = 5000
_COMPLETED_NON_STREAMING_REQUEST_ID_LIMIT = 10000
_COMPLETED_NON_STREAMING_REQUEST_ID_RETAINED = 5000
# A stream_done waiting on one of its own unprocessed chunks must not hang the
# request forever when that chunk never lands: after this bound it completes
# anyway (with a warning), dropping the missing chunk.
_STREAM_DONE_MAX_WAIT_S = 5.0


class StreamingSimpleScheduler:
    """Scheduler base for simple stages with streaming input.

    Subclasses implement the streaming hooks. Non-streaming requests use
    ``compute_fn`` / ``batch_compute_fn`` exactly like SimpleScheduler, while
    streaming requests are kept out of the non-streaming batch path.
    """

    _can_batch_stream_chunks: bool = False
    _stream_chunk_batch_max: int | None = None
    _stream_chunk_batch_distinct_requests: bool = False

    def __init__(
        self,
        compute_fn: Callable[[Any], Any] | None,
        *,
        batch_compute_fn: Callable[[list[Any]], list[Any]] | None = None,
        max_batch_size: int = 1,
        max_batch_wait_ms: int = 0,
        request_cost_fn: Callable[[Any], int] | None = None,
        max_batch_cost: int | None = None,
        abort_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self.requires_tp_work_fanout: bool = True

        self._fn = compute_fn
        self._batch_fn = batch_compute_fn
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._request_cost_fn = request_cost_fn
        self._max_batch_cost = (
            max(int(max_batch_cost), 0) if max_batch_cost is not None else None
        )
        self._abort_callback = abort_callback

        self._running = False
        self._pending_messages: collections.deque[IncomingMessage] = collections.deque()
        self._pending_done: set[str] = set()
        self._deferred_done_deadlines: dict[str, float] = {}
        self._stream_payloads: dict[str, Any] = {}
        self._aborted_request_ids: set[str] = set()
        self._completed_non_streaming_request_ids: set[str] = set()
        self._state_lock = threading.RLock()
        self._abort_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def is_streaming_payload(self, payload: Any) -> bool:
        return False

    def validate_non_streaming_payload(self, payload: Any) -> None:
        del payload

    def on_streaming_new_request(self, request_id: str, payload: Any) -> None:
        del request_id, payload

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        del request_id, item
        return []

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        """Caller holds no lock and ignores any return.

        Subclasses own their locking and emit via outbox internally.
        """
        for request_id, item in items:
            if self._is_aborted(request_id):
                continue
            try:
                self._handle_stream_chunk(request_id, item)
            except Exception as exc:
                self._emit_error(request_id, exc)
                self.abort(request_id)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        del request_id
        return []

    def on_stream_done_before_payload(self, request_id: str) -> list[OutgoingMessage]:
        """Output a stage owes the client the moment the producer signals EOS.

        ``stream_done`` routinely lands before the terminal payload, so the
        rest of ``on_stream_done`` is deferred until the payload latches the
        request. Anything already computed must not wait on that latch.
        """
        del request_id
        return []

    def clear_stream_state(self, request_id: str) -> None:
        del request_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        loop = asyncio.new_event_loop()
        try:
            while self._running:
                msg = self._next_message()
                if msg is None:
                    # No work and no message: give deferred dones their
                    # deadline check so a missing chunk cannot stall forever.
                    self._resume_deferred_stream_done()
                    continue
                if self._is_aborted(msg.request_id):
                    continue
                try:
                    self._handle_message(msg, loop)
                except Exception as exc:
                    logger.exception(
                        "%s failed for %s",
                        self.__class__.__name__,
                        msg.request_id,
                    )
                    self._emit_error(msg.request_id, exc)
                    self.abort(msg.request_id)
        finally:
            loop.close()

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._abort_state(request_id)
        self._cleanup_aborted_request(request_id)

    def _abort_state(self, request_id: str) -> None:
        self._record_aborted_request_id(request_id)
        self._clear_request_state(request_id, keep_aborted=True)

    def _handle_message(
        self, msg: IncomingMessage, loop: asyncio.AbstractEventLoop
    ) -> None:
        if msg.type == "new_request":
            self._handle_new_request_batch(self._collect_new_request_batch(msg), loop)
            return
        if msg.type == "stream_chunk":
            if self._can_batch_stream_chunks:
                self._handle_stream_chunk_batch(self._collect_stream_chunk_batch(msg))
            else:
                self._on_chunk(msg.request_id, msg.data)
            return
        if msg.type == "stream_done":
            self._on_done(msg.request_id)
            return
        raise ValueError(f"Unsupported streaming scheduler message type: {msg.type}")

    def _next_message(self) -> IncomingMessage | None:
        if self._pending_messages:
            return self._pending_messages.popleft()
        try:
            return self.inbox.get(timeout=0.1)
        except _queue_mod.Empty:
            return None

    def _defer_message(self, msg: IncomingMessage) -> None:
        """Put a message back into the deferred queue in arrival order.

        Front-inserting a deferral makes the queue LIFO, so the newest deferred
        message is served first and the oldest can be postponed indefinitely.
        A message stamped with ``arrived_at`` (the inbox instrumentation) is
        placed before the first newer one; without the stamp it is appended,
        which keeps arrival order because it was popped before anything still
        queued could arrive."""
        arrived_at = getattr(msg, "arrived_at", None)
        if arrived_at is None:
            self._pending_messages.append(msg)
            return
        for index, queued in enumerate(self._pending_messages):
            queued_at = getattr(queued, "arrived_at", None)
            if queued_at is not None and queued_at > arrived_at:
                self._pending_messages.insert(index, msg)
                return
        self._pending_messages.append(msg)

    # ------------------------------------------------------------------
    # Abort and cleanup
    # ------------------------------------------------------------------

    def _record_aborted_request_id(self, request_id: str) -> None:
        with self._abort_lock:
            self._aborted_request_ids.add(request_id)
            if len(self._aborted_request_ids) <= _ABORTED_REQUEST_ID_LIMIT:
                return
            excess = len(self._aborted_request_ids) - _ABORTED_REQUEST_ID_RETAINED
            for stale_request_id in list(self._aborted_request_ids)[:excess]:
                self._aborted_request_ids.discard(stale_request_id)

    def _is_aborted(self, request_id: str) -> bool:
        with self._abort_lock:
            return request_id in self._aborted_request_ids

    def _clear_request_state(
        self, request_id: str, *, keep_aborted: bool = False
    ) -> None:
        with self._state_lock:
            self._stream_payloads.pop(request_id, None)
            self._pending_done.discard(request_id)
            self._deferred_done_deadlines.pop(request_id, None)
            self.clear_stream_state(request_id)
            if not keep_aborted:
                with self._abort_lock:
                    self._aborted_request_ids.discard(request_id)

    def _record_completed_non_streaming_request_id(self, request_id: str) -> None:
        with self._state_lock:
            self._completed_non_streaming_request_ids.add(request_id)
            if (
                len(self._completed_non_streaming_request_ids)
                <= _COMPLETED_NON_STREAMING_REQUEST_ID_LIMIT
            ):
                return
            excess = (
                len(self._completed_non_streaming_request_ids)
                - _COMPLETED_NON_STREAMING_REQUEST_ID_RETAINED
            )
            for stale_request_id in list(self._completed_non_streaming_request_ids)[
                :excess
            ]:
                self._completed_non_streaming_request_ids.discard(stale_request_id)

    def _cleanup_aborted_request(self, request_id: str) -> None:
        if self._abort_callback is None:
            return
        try:
            self._abort_callback(request_id)
        except Exception:
            logger.exception(
                "%s: abort cleanup failed for %s",
                self.__class__.__name__,
                request_id,
            )

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    def _message_cost(self, msg: IncomingMessage) -> int:
        if self._request_cost_fn is None or msg.type != "new_request":
            return 0
        return max(int(self._request_cost_fn(msg.data)), 0)

    def _collect_new_request_batch(
        self, first_msg: IncomingMessage
    ) -> list[IncomingMessage]:
        batch = [first_msg]
        if (
            self._batch_fn is None
            or self._max_batch_size <= 1
            or self.is_streaming_payload(first_msg.data)
        ):
            return batch

        batch_cost = self._message_cost(first_msg)
        deadline = time.monotonic() + self._max_batch_wait_s
        while len(batch) < self._max_batch_size:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self.inbox.get(timeout=remaining)
                except _queue_mod.Empty:
                    break

            if self._is_aborted(msg.request_id):
                continue
            if msg.type != "new_request":
                if (
                    msg.type == "stream_done"
                    and msg.request_id not in self._stream_payloads
                ):
                    # Note(Chenchen Hong): Done-before-payload only latches state,
                    # so defer it and keep looking for terminal payloads that can batch.
                    self._pending_messages.append(msg)
                    continue
                self._pending_messages.append(msg)
                break
            try:
                is_streaming = self.is_streaming_payload(msg.data)
            except Exception as exc:
                self._emit_error(msg.request_id, exc)
                self.abort(msg.request_id)
                continue
            if is_streaming:
                self._pending_messages.append(msg)
                break
            if self._max_batch_cost is not None:
                try:
                    msg_cost = self._message_cost(msg)
                except Exception as exc:
                    self._emit_error(msg.request_id, exc)
                    self.abort(msg.request_id)
                    continue
                if batch and batch_cost + msg_cost > self._max_batch_cost:
                    self._defer_message(msg)
                    break
                batch_cost += msg_cost
            batch.append(msg)
        return batch

    def _collect_stream_chunk_batch(
        self, first_msg: IncomingMessage
    ) -> list[IncomingMessage]:
        """Pushback of the first non-chunk message in arrival order; no blocking
        wait, so only already-queued chunks coalesce."""
        batch = [first_msg]
        seen_request_ids = (
            {first_msg.request_id}
            if self._stream_chunk_batch_distinct_requests
            else None
        )
        cap = self._stream_chunk_batch_max or max(self._max_batch_size, 1)
        if cap <= 1:
            return batch
        while len(batch) < cap:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                break
            if msg.type != "stream_chunk":
                self._defer_message(msg)
                break
            if self._is_aborted(msg.request_id):
                continue
            if seen_request_ids is not None and msg.request_id in seen_request_ids:
                self._defer_message(msg)
                break
            batch.append(msg)
            if seen_request_ids is not None:
                seen_request_ids.add(msg.request_id)
        return batch

    def _handle_new_request_batch(
        self,
        batch: list[IncomingMessage],
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        owns_loop = loop is None
        if loop is None:
            loop = asyncio.new_event_loop()
        try:
            self._handle_new_request_batch_with_loop(batch, loop)
        finally:
            if owns_loop:
                loop.close()

    def _handle_new_request_batch_with_loop(
        self,
        batch: list[IncomingMessage],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        streaming: list[IncomingMessage] = []
        non_streaming: list[IncomingMessage] = []
        for msg in batch:
            try:
                is_streaming = self.is_streaming_payload(msg.data)
            except Exception as exc:
                self._emit_error(msg.request_id, exc)
                self.abort(msg.request_id)
                continue
            if is_streaming:
                streaming.append(msg)
            else:
                non_streaming.append(msg)

        for msg in streaming:
            if self._is_aborted(msg.request_id):
                continue
            self._handle_streaming_new_request(msg.request_id, msg.data)

        if non_streaming:
            self._run_non_streaming_batch(non_streaming, loop)

    def _run_non_streaming_batch(
        self,
        batch: list[IncomingMessage],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        active = [msg for msg in batch if not self._is_aborted(msg.request_id)]
        if not active:
            return
        with self._state_lock:
            for msg in active:
                self._pending_done.discard(msg.request_id)
                self._deferred_done_deadlines.pop(msg.request_id, None)

        valid: list[IncomingMessage] = []
        for msg in active:
            try:
                self.validate_non_streaming_payload(msg.data)
            except Exception as exc:
                self._emit_error(msg.request_id, exc)
                self._record_completed_non_streaming_request_id(msg.request_id)
                continue
            valid.append(msg)
        if not valid:
            return

        if self._batch_fn is None or len(valid) <= 1:
            for msg in valid:
                if self._is_aborted(msg.request_id):
                    continue
                try:
                    result = self._run_compute(msg.data, loop)
                except Exception as exc:
                    if not self._is_aborted(msg.request_id):
                        self._emit_error(msg.request_id, exc)
                        self._record_completed_non_streaming_request_id(msg.request_id)
                    continue
                if not self._is_aborted(msg.request_id):
                    self._emit_result(msg.request_id, result)
                    self._record_completed_non_streaming_request_id(msg.request_id)
            return

        try:
            results = self._batch_fn([msg.data for msg in valid])
            if asyncio.iscoroutine(results):
                results = loop.run_until_complete(results)
        except Exception as exc:
            for msg in valid:
                if not self._is_aborted(msg.request_id):
                    self._emit_error(msg.request_id, exc)
                    self._record_completed_non_streaming_request_id(msg.request_id)
            return
        if len(results) != len(valid):
            exc = ValueError(
                f"batch_compute_fn returned {len(results)} results for "
                f"{len(valid)} requests"
            )
            for msg in valid:
                if not self._is_aborted(msg.request_id):
                    self._emit_error(msg.request_id, exc)
                    self._record_completed_non_streaming_request_id(msg.request_id)
            return
        for msg, result in zip(valid, results):
            if not self._is_aborted(msg.request_id):
                self._emit_result(msg.request_id, result)
                self._record_completed_non_streaming_request_id(msg.request_id)

    def _run_compute(self, payload: Any, loop: asyncio.AbstractEventLoop) -> Any:
        if self._fn is None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not support non-streaming compute"
            )
        result = self._fn(payload)
        if asyncio.iscoroutine(result):
            result = loop.run_until_complete(result)
        return result

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    def _validate_stream_chunk_item(self, request_id: str, item: Any) -> StreamItem:
        if not isinstance(item, StreamItem):
            raise TypeError(
                f"{self.__class__.__name__} expected StreamItem for "
                f"{request_id!r}, got {type(item).__name__}"
            )
        return item

    def _handle_streaming_new_request(self, request_id: str, payload: Any) -> None:
        with self._abort_lock:
            self._aborted_request_ids.discard(request_id)
        replay_done = False
        with self._state_lock:
            self._completed_non_streaming_request_ids.discard(request_id)
            self._stream_payloads[request_id] = payload
            self.on_streaming_new_request(request_id, payload)
            if request_id in self._pending_done:
                self._pending_done.discard(request_id)
                replay_done = True
        if replay_done:
            # Off ``_state_lock``: the replayed done may run the external abort
            # cleanup, which must never hold the lock.
            self._handle_stream_done(request_id)

    def _handle_stream_chunk(self, request_id: str, item: Any) -> None:
        item = self._validate_stream_chunk_item(request_id, item)
        with self._state_lock:
            for out in self.on_stream_chunk(request_id, item):
                if not self._is_aborted(request_id):
                    self.outbox.put(out)
        # Off ``_state_lock`` so a failing resume's abort callback is not run
        # under it (the batch path is shaped the same way).
        self._resume_deferred_stream_done()

    def _handle_stream_chunk_batch(self, batch: list[IncomingMessage]) -> None:
        items: list[tuple[str, StreamItem]] = []
        for msg in batch:
            if self._is_aborted(msg.request_id):
                continue
            try:
                item = self._validate_stream_chunk_item(msg.request_id, msg.data)
            except Exception as exc:
                self._emit_error(msg.request_id, exc)
                self.abort(msg.request_id)
                continue
            items.append((msg.request_id, item))
        items = [
            (request_id, item)
            for request_id, item in items
            if not self._is_aborted(request_id)
        ]
        if items:
            self.on_stream_chunk_batch(items)
        self._resume_deferred_stream_done()

    def _handle_stream_done(self, request_id: str) -> None:
        drain_failure: BaseException | None = None
        with self._state_lock:
            if self._is_aborted(request_id):
                # An abort already owns this request's error and cleanup.
                return
            if request_id not in self._stream_payloads:
                if request_id in self._completed_non_streaming_request_ids:
                    return
                self._pending_done.add(request_id)
                for out in self.on_stream_done_before_payload(request_id):
                    if not self._is_aborted(request_id):
                        self.outbox.put(out)
                return
            try:
                self.drain_parked_chunks_for_done(request_id)
            except Exception as exc:
                # Exactly one owner: error and abort the request here, and run
                # its external cleanup below, off ``_state_lock``.
                drain_failure = exc
                self._emit_error(request_id, exc)
                self._abort_state(request_id)
                self._drop_deferred_stream_done(request_id)
            if drain_failure is None and self._defer_stream_done_for_pending_chunk(
                request_id
            ):
                return
            # The drain above failed, or an abort landed while the done was
            # judged: never finalize a request that is no longer live.
            if not self._is_aborted(request_id) and request_id in self._stream_payloads:
                for out in self.on_stream_done(request_id):
                    if not self._is_aborted(request_id):
                        self.outbox.put(out)
                if not self._is_aborted(request_id):
                    self._clear_request_state(request_id)
        if drain_failure is not None:
            self._cleanup_aborted_request(request_id)

    def drain_parked_chunks_for_done(self, request_id: str) -> None:
        """Hook: ingest this request's parked chunks before its done is judged.

        A chunk parked by a collector can sit behind a long pump step, which
        outlives any sane wait bound. The chunk is already in hand, so the
        default scheduler has nothing to do; vocoders that ingest chunks pull
        their own parked ones in here. An implementation raises when a parked
        chunk cannot be ingested: ``_handle_stream_done`` owns the request's
        single error and abort, and its external cleanup, off ``_state_lock``."""

    def _stream_has_unprocessed_chunk(self, request_id: str) -> bool:
        """Whether this request still owns a chunk that has not been ingested.

        Ordering is request-local: a chunk parked for another request must not
        hold up this done. Subclasses that also buffer chunks outside the
        shared queues (see ``StreamingVocoderBase``) extend this."""
        return any(
            msg.type == "stream_chunk" and msg.request_id == request_id
            for msg in self._pending_messages
        )

    def _defer_stream_done_for_pending_chunk(self, request_id: str) -> bool:
        """True while the request's own chunk is still unprocessed: completing
        now would clear the state and drop that chunk. The wait is bounded so a
        chunk id that never arrives cannot hang the request."""
        if not self._stream_has_unprocessed_chunk(request_id):
            self._drop_deferred_stream_done(request_id)
            return False
        deadline = self._deferred_done_deadlines.setdefault(
            request_id, time.monotonic() + _STREAM_DONE_MAX_WAIT_S
        )
        self._pending_done.add(request_id)
        if time.monotonic() < deadline:
            return True
        logger.warning(
            "%s: stream_done for %s waited %.1fs on an unprocessed chunk; "
            "completing without it",
            self.__class__.__name__,
            request_id,
            _STREAM_DONE_MAX_WAIT_S,
        )
        self._drop_deferred_stream_done(request_id)
        return False

    def _drop_deferred_stream_done(self, request_id: str) -> None:
        self._pending_done.discard(request_id)
        self._deferred_done_deadlines.pop(request_id, None)

    def _resume_deferred_stream_done(self) -> None:
        """Re-dispatch done markers whose own chunks have landed (or whose wait
        bound expired). Runs after chunk delivery and whenever the serving loop
        goes idle, so a deferred done never depends on a later arrival.

        A failing final decode stays with its own request: the resume is
        triggered by another request's chunk (or by the idle loop), so letting
        the exception escape would abort the innocent request or kill the
        serving loop and re-raise on every later resume."""
        with self._state_lock:
            request_ids = list(self._deferred_done_deadlines)
        for request_id in request_ids:
            try:
                self._handle_stream_done(request_id)
            except Exception as exc:
                logger.exception(
                    "%s: deferred stream_done failed for %s",
                    self.__class__.__name__,
                    request_id,
                )
                with self._state_lock:
                    self._emit_error(request_id, exc)
                    self._abort_state(request_id)
                    self._drop_deferred_stream_done(request_id)
                self._cleanup_aborted_request(request_id)

    # Compatibility wrappers for existing tests and subclasses.
    def _on_streaming_new_request(self, request_id: str, payload: Any) -> None:
        self._handle_streaming_new_request(request_id, payload)

    def _on_chunk(self, request_id: str, item: StreamItem) -> None:
        self._handle_stream_chunk(request_id, item)

    def _on_done(self, request_id: str) -> None:
        self._handle_stream_done(request_id)

    # ------------------------------------------------------------------
    # Outbox helpers
    # ------------------------------------------------------------------

    def _emit_result(self, request_id: str, result: Any) -> None:
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=result,
            )
        )

    def _emit_error(self, request_id: str, error: BaseException) -> None:
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="error",
                data=error,
            )
        )
