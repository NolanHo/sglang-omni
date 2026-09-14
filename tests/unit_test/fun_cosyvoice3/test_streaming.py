# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import EOS_ID, VOCAB_SIZE
from sglang_omni.models.fun_cosyvoice3.streaming import (
    AR_FOLLOWUP_FLUSH_TOKENS,
    AR_INITIAL_FLUSH_TOKENS,
    LEFTOVER_FLOW_STREAMING,
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MEL_RATIO,
    first_ar_flush_tokens,
    next_stream_hop_len,
    pad_flow_prompt_to_hop,
    prompt_token_pad,
    stream_hop_len,
    tokens_needed_for_causal_chunk,
)
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
    FunCosyVoice3StreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling import streaming_simple_scheduler
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import _STREAM_DONE_MAX_WAIT_S


def test_stream_hop_math_matches_cosyvoice3() -> None:
    assert prompt_token_pad(0) == 0
    assert prompt_token_pad(10) == 15
    assert prompt_token_pad(25) == 0
    assert prompt_token_pad(26) == 24
    assert stream_hop_len(0, hop_len=25, prompt_pad=15) == 40
    assert stream_hop_len(40, hop_len=25, prompt_pad=15) == 25
    assert next_stream_hop_len(25) == 50
    assert next_stream_hop_len(50) == 100
    assert next_stream_hop_len(100) == 100
    assert next_stream_hop_len(25, max_hop_len=50) == 50
    assert next_stream_hop_len(50, max_hop_len=50) == 50
    assert next_stream_hop_len(25, disable_growth=True) == 25
    assert tokens_needed_for_causal_chunk(0, hop_len=25, prompt_pad=0) == 28
    assert tokens_needed_for_causal_chunk(0, hop_len=25, prompt_pad=15) == 43
    assert first_ar_flush_tokens(0) == AR_INITIAL_FLUSH_TOKENS
    assert first_ar_flush_tokens(10) == AR_INITIAL_FLUSH_TOKENS
    assert first_ar_flush_tokens(25) == AR_INITIAL_FLUSH_TOKENS
    assert first_ar_flush_tokens(0, hop_len=15) == 18


def test_pad_flow_prompt_repeats_last_frame_to_hop_multiple() -> None:
    token = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.int32)
    feat = (
        torch.arange(10 * TOKEN_MEL_RATIO, dtype=torch.float32)
        .reshape(1, 10 * TOKEN_MEL_RATIO, 1)
        .repeat(1, 1, 80)
    )
    padded_token, padded_feat = pad_flow_prompt_to_hop(token, feat)
    assert tuple(padded_token.shape) == (1, 25)
    assert torch.equal(padded_token[:, :10], token)
    assert torch.equal(padded_token[:, 10:], torch.full((1, 15), 10, dtype=torch.int32))
    assert tuple(padded_feat.shape) == (1, 50, 80)
    assert torch.equal(padded_feat[:, :20], feat)
    assert torch.equal(padded_feat[:, 20:], feat[:, -1:, :].repeat(1, 30, 1))
    aligned_token, aligned_feat = pad_flow_prompt_to_hop(padded_token, padded_feat)
    assert torch.equal(aligned_token, padded_token)
    assert torch.equal(aligned_feat, padded_feat)


class _FakeEstimator(torch.nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("causal hops must use CosyVoice Flow.inference")


class _FakeFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[dict] = []
        self.decoder = SimpleNamespace(estimator=_FakeEstimator())

    def inference(self, **kwargs):
        self.calls.append(kwargs)
        token_count = int(kwargs["token"].shape[1])
        if not kwargs.get("finalize", True):
            token_count = max(token_count - PRE_LOOKAHEAD_LEN, 0)
        return torch.ones(1, 80, token_count * TOKEN_MEL_RATIO), None


class _FakeHiFT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[tuple] = []
        self.upsample_rates = [8, 5, 3]
        self.istft_params = {"n_fft": 16, "hop_len": 4}

    def inference(self, *, speech_feat, finalize):
        self.calls.append((speech_feat, finalize))
        return torch.arange(speech_feat.shape[-1]).reshape(1, -1).float(), None


def _drain(scheduler: FunCosyVoice3StreamingVocoderScheduler) -> list[OutgoingMessage]:
    messages: list[OutgoingMessage] = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except Empty:
            return messages


def _drain_inbox(
    scheduler: FunCosyVoice3StreamingVocoderScheduler,
) -> list[IncomingMessage]:
    messages: list[IncomingMessage] = []
    while True:
        try:
            messages.append(scheduler.inbox.get_nowait())
        except Empty:
            return messages


def _waveform(data: dict) -> np.ndarray:
    return np.frombuffer(data["audio_waveform"], dtype=np.float32).reshape(
        data["audio_waveform_shape"]
    )


def _scheduler(
    **scheduler_kwargs,
) -> tuple[_FakeFlow, FunCosyVoice3StreamingVocoderScheduler]:
    flow = _FakeFlow()
    return flow, FunCosyVoice3StreamingVocoderScheduler(
        stages.CosyVoice3Vocoder(flow, _FakeHiFT()),
        **scheduler_kwargs,
    )


def _model_runner() -> FunCosyVoice3ModelRunner:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner._token_hop_len = TOKEN_HOP_LEN
    runner._ar_followup_flush_tokens = AR_FOLLOWUP_FLUSH_TOKENS
    runner._outbox = Queue()
    runner._vocoder_target = "vocoder"
    return runner


def _stream_payload(
    request_id: str = "req-stream",
    *,
    codes: list[int] | None = None,
    prompt_feat_frames: int = 0,
    prompt_token_len: int = 0,
) -> StagePayload:
    state = FunCosyVoice3State(
        text="hello",
        stream=True,
        audio_codes=None if codes is None else torch.tensor(codes, dtype=torch.long),
        flow_prompt_speech_token=torch.zeros(1, prompt_token_len, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, prompt_feat_frames, 80),
        flow_embedding=torch.ones(1, 192),
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": True}),
        data=state.to_dict(),
    )


def _item(tokens: list[int], *, chunk_id: int = 0) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=torch.tensor(tokens, dtype=torch.long),
        from_stage="tts_engine",
        metadata={"modality": "audio_codes", "stream": True},
    )


def _chunk_message(
    request_id: str, tokens: list[int], *, chunk_id: int
) -> IncomingMessage:
    return IncomingMessage(
        request_id=request_id,
        type="stream_chunk",
        data=_item(tokens, chunk_id=chunk_id),
    )


def _done_message(request_id: str) -> IncomingMessage:
    return IncomingMessage(request_id=request_id, type="stream_done")


def _run_serving_loop(
    scheduler: FunCosyVoice3StreamingVocoderScheduler, *, limit: int = 32
) -> None:
    """Dispatch queued messages the way the serving loop does.

    Bounded so a message the scheduler keeps re-queueing fails the calling
    assertion instead of hanging the test session.
    """
    for _ in range(limit):
        if not scheduler._pending_messages and scheduler.inbox.empty():
            return
        message = scheduler._next_message()
        assert message is not None, "scheduler returned no message with work pending"
        scheduler._handle_message(message, None)
    raise AssertionError("scheduler did not settle; a queued message livelocked")


class _FakeClock:
    """Stand-in for the scheduler module's ``time``, so a test can pass the
    stream-done wait bound without sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _serve(
    scheduler: FunCosyVoice3StreamingVocoderScheduler, *, output_count: int
) -> list[OutgoingMessage]:
    """Run the real serving loop in a thread and collect ``output_count``
    outbox messages; ``queue.Empty`` means the loop stopped early."""
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        return [scheduler.outbox.get(timeout=2.0) for _ in range(output_count)]
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)


def _resume_without_raising(
    scheduler: FunCosyVoice3StreamingVocoderScheduler,
) -> list[BaseException]:
    """Resume deferred dones, returning whatever escaped instead of raising."""
    raised: list[BaseException] = []
    try:
        scheduler._resume_deferred_stream_done()
    except BaseException as exc:  # noqa: BLE001 - the escape is the assertion
        raised.append(exc)
    return raised


def test_streaming_vocoder_emits_causal_chunk_then_finalizes_remainder() -> None:
    flow, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-stream", _stream_payload())
    scheduler._on_chunk("req-stream", _item(list(range(28))))
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream"]
    assert flow.calls[0]["streaming"] is True
    assert flow.calls[0]["finalize"] is False
    assert int(flow.calls[0]["token"].shape[1]) == 28
    assert _waveform(messages[0].data).shape == (50,)

    scheduler._on_done("req-stream")
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "result"]
    assert LEFTOVER_FLOW_STREAMING is False
    assert flow.calls[1]["streaming"] is False
    assert flow.calls[1]["finalize"] is True
    assert _waveform(messages[0].data).shape == (6,)
    assert messages[1].data.data["modality"] == "audio"
    assert messages[1].data.data["sample_rate"] == 24000


def test_streaming_vocoder_does_not_decode_before_lookahead_tokens_arrive() -> None:
    flow, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-stream", _stream_payload())
    scheduler._on_chunk("req-stream", _item(list(range(27))))
    assert _drain(scheduler) == []
    assert flow.calls == []


def test_streaming_vocoder_pads_prompt_and_decodes_first_hop_at_28() -> None:
    flow, scheduler = _scheduler()
    prompt_len = 10
    scheduler._on_streaming_new_request(
        "req-pad",
        _stream_payload(
            "req-pad",
            prompt_token_len=prompt_len,
            prompt_feat_frames=prompt_len * TOKEN_MEL_RATIO,
        ),
    )
    scheduler._on_chunk("req-pad", _item(list(range(27))))
    assert _drain(scheduler) == []
    assert flow.calls == []

    scheduler._on_chunk("req-pad", _item([27], chunk_id=1))
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream"]
    assert int(flow.calls[0]["prompt_token"].shape[1]) == 25
    assert int(flow.calls[0]["prompt_feat"].shape[1]) == 50
    assert int(flow.calls[0]["token"].shape[1]) == 28
    assert flow.calls[0]["streaming"] is True
    assert flow.calls[0]["finalize"] is False
    assert _waveform(messages[0].data).shape == (TOKEN_HOP_LEN * TOKEN_MEL_RATIO,)


def test_model_runner_flushes_speech_tokens_and_skips_control_ids() -> None:
    runner = _model_runner()
    data = CosyVoice3SGLangRequestData(
        stream_metadata={"modality": "audio_codes", "stream": True},
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 0, 80),
        flow_embedding=torch.ones(1, 192),
    )
    request = SimpleNamespace(request_id="req-ar", data=data)

    for token_id in range(AR_INITIAL_FLUSH_TOKENS - 1):
        runner._collect_tokens(
            SimpleNamespace(next_token_ids=torch.tensor([token_id])),
            None,
            None,
            [request],
        )
    assert runner._outbox.empty()

    runner._collect_tokens(
        SimpleNamespace(next_token_ids=torch.tensor([EOS_ID])),
        None,
        None,
        [request],
    )
    assert runner._outbox.empty()
    assert all(code.item() < VOCAB_SIZE for code in data.output_codes)

    runner._collect_tokens(
        SimpleNamespace(next_token_ids=torch.tensor([7])),
        None,
        None,
        [request],
    )
    message = runner._outbox.get_nowait()
    assert message.type == "stream"
    assert message.target == "vocoder"
    assert message.metadata["stream"] is True
    assert message.metadata["flow_embedding"].shape == (1, 192)
    assert tuple(message.data.tolist()) == tuple(range(AR_INITIAL_FLUSH_TOKENS - 1)) + (
        7,
    )
    assert data.stream_code_next_flush == (
        AR_INITIAL_FLUSH_TOKENS + AR_FOLLOWUP_FLUSH_TOKENS
    )

    runner.on_request_finished("req-ar", data)
    assert runner._outbox.empty()


def test_model_runner_first_flush_ignores_prompt_pad() -> None:
    runner = _model_runner()
    prompt_len = 10
    data = CosyVoice3SGLangRequestData(
        stream_metadata={"modality": "audio_codes", "stream": True},
        flow_prompt_speech_token=torch.zeros(1, prompt_len, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 1, 80),
        flow_embedding=torch.ones(1, 192),
    )
    request = SimpleNamespace(request_id="req-pad", data=data)
    first_flush = first_ar_flush_tokens(prompt_len)
    assert first_flush == AR_INITIAL_FLUSH_TOKENS

    early = _feed_tokens(runner, request, list(range(first_flush - 1)))
    assert early == []
    assert data.stream_code_next_flush == first_flush

    ready = _feed_tokens(runner, request, [7])
    assert len(ready) == 1
    assert tuple(ready[0].data.tolist()) == tuple(range(first_flush - 1)) + (7,)
    assert data.stream_code_next_flush == first_flush + AR_FOLLOWUP_FLUSH_TOKENS


def _feed_tokens(
    runner: FunCosyVoice3ModelRunner,
    request: SimpleNamespace,
    token_ids: list[int],
) -> list[OutgoingMessage]:
    for token_id in token_ids:
        runner._collect_tokens(
            SimpleNamespace(next_token_ids=torch.tensor([token_id])),
            None,
            None,
            [request],
        )
    messages: list[OutgoingMessage] = []
    while True:
        try:
            messages.append(runner._outbox.get_nowait())
        except Empty:
            return messages


def _to_stream_chunk(outgoing: OutgoingMessage, chunk_id: int) -> IncomingMessage:
    return IncomingMessage(
        request_id=outgoing.request_id,
        type="stream_chunk",
        data=StreamItem(
            chunk_id=chunk_id,
            data=outgoing.data,
            from_stage="tts_engine",
            metadata=outgoing.metadata,
        ),
    )


def test_ar_to_vocoder_grows_hops_then_finalizes_remainder() -> None:
    request_id = "req-hops"
    runner = _model_runner()
    # note (guozhihao-224): feat uses a non-empty time axis because (1, 0, 80)
    # does not round-trip through the tensor_list wire codec.
    data = CosyVoice3SGLangRequestData(
        stream_metadata={"modality": "audio_codes", "stream": True},
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 1, 80),
        flow_embedding=torch.ones(1, 192),
    )
    request = SimpleNamespace(request_id=request_id, data=data)
    flow, scheduler = _scheduler()

    generated = list(range(AR_INITIAL_FLUSH_TOKENS + 2 * AR_FOLLOWUP_FLUSH_TOKENS))
    ar_messages = _feed_tokens(runner, request, generated)
    assert len(ar_messages) == 3
    assert "flow_embedding" in ar_messages[0].metadata
    assert "flow_embedding" not in ar_messages[1].metadata

    pcm_chunks: list[np.ndarray] = []
    for chunk_id, outgoing in enumerate(ar_messages):
        scheduler._handle_message(_to_stream_chunk(outgoing, chunk_id), None)
        for message in _drain(scheduler):
            assert message.type == "stream"
            pcm_chunks.append(_waveform(message.data))

    assert [int(call["token"].shape[1]) for call in flow.calls] == [28, 78]
    assert all(
        call["streaming"] is True and call["finalize"] is False for call in flow.calls
    )
    assert [chunk.shape[0] for chunk in pcm_chunks] == [
        TOKEN_HOP_LEN * TOKEN_MEL_RATIO,
        2 * TOKEN_HOP_LEN * TOKEN_MEL_RATIO,
    ]

    scheduler._handle_message(
        IncomingMessage(request_id=request_id, type="stream_done"), None
    )
    assert _drain(scheduler) == []

    scheduler._handle_message(
        IncomingMessage(
            request_id=request_id,
            type="new_request",
            data=_stream_payload(request_id, codes=generated, prompt_feat_frames=1),
        ),
        None,
    )
    final_messages = _drain(scheduler)
    assert [message.type for message in final_messages] == ["stream", "result"]
    remainder = _waveform(final_messages[0].data)
    assert remainder.shape == (PRE_LOOKAHEAD_LEN * TOKEN_MEL_RATIO,)
    assert flow.calls[-1]["streaming"] is False
    assert flow.calls[-1]["finalize"] is True
    total = np.concatenate(pcm_chunks + [remainder])
    assert total.shape == (len(generated) * TOKEN_MEL_RATIO,)


def test_streaming_vocoder_fallback_raises_on_empty_audio_codes() -> None:
    _, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-empty", _stream_payload(codes=None))
    with pytest.raises(RuntimeError, match="no usable speech tokens"):
        scheduler._on_done("req-empty")

    _, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-empty-list", _stream_payload(codes=[]))
    with pytest.raises(RuntimeError, match="no usable speech tokens"):
        scheduler._on_done("req-empty-list")


def test_streaming_vocoder_enables_first_hop_coalescing_by_default() -> None:
    _, scheduler = _scheduler()
    assert scheduler._can_batch_stream_chunks is True
    assert scheduler._stream_chunk_batch_distinct_requests is True
    assert scheduler._first_hop_peer_wait_ms == 30
    assert scheduler._can_batch_follow_up_hops is True


def test_equal_first_hops_share_one_causal_flow_batch() -> None:
    from sglang_omni.models.fun_cosyvoice3.stages import FunCosyVoice3Flow
    from tests.unit_test.fun_cosyvoice3.test_flow_batch import _FakeFlow as _PackedFlow

    flow = _PackedFlow(channels=80, max_frames=128)
    flow.spk_embed_affine_layer = torch.nn.Linear(192, 80, bias=False)
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        stages.CosyVoice3Vocoder(FunCosyVoice3Flow(flow), _FakeHiFT()),
        max_batch_size=8,
    )
    scheduler._can_batch_stream_chunks = True
    # note (guozhihao-224): empty (1, 0, 80) prompt_feat round-trips through
    # tensor_list as [[]] and loses the channel dim; use a hop-aligned prompt.
    prompt_len = TOKEN_HOP_LEN
    payload_kwargs = {
        "prompt_token_len": prompt_len,
        "prompt_feat_frames": prompt_len * TOKEN_MEL_RATIO,
    }
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(
            request_id, _stream_payload(request_id, **payload_kwargs)
        )
    for request_id in ("req-a", "req-b"):
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert flow.decoder.estimator.calls
    assert flow.decoder.estimator.calls[0]["streaming"] is True
    assert flow.decoder.estimator.calls[0]["x"].shape[0] == 4
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "stream"]
    assert {_waveform(message.data).shape[0] for message in messages} == {
        TOKEN_HOP_LEN * TOKEN_MEL_RATIO
    }


def test_disabled_coalescing_keeps_equal_first_hops_serial() -> None:
    flow, scheduler = _scheduler()
    scheduler._can_batch_stream_chunks = False
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(request_id, _stream_payload(request_id))
    for request_id in ("req-a", "req-b"):
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [28, 28]
    assert all(
        call["streaming"] is True and call["finalize"] is False for call in flow.calls
    )


def test_late_payloads_share_one_causal_flow_batch() -> None:
    from sglang_omni.models.fun_cosyvoice3.stages import FunCosyVoice3Flow
    from tests.unit_test.fun_cosyvoice3.test_flow_batch import _FakeFlow as _PackedFlow

    flow = _PackedFlow(channels=80, max_frames=128)
    flow.spk_embed_affine_layer = torch.nn.Linear(192, 80, bias=False)
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        stages.CosyVoice3Vocoder(FunCosyVoice3Flow(flow), _FakeHiFT()),
        max_batch_size=8,
    )
    scheduler._can_batch_stream_chunks = True
    prompt_len = TOKEN_HOP_LEN
    payload_kwargs = {
        "prompt_token_len": prompt_len,
        "prompt_feat_frames": prompt_len * TOKEN_MEL_RATIO,
    }
    for request_id in ("req-a", "req-b"):
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
        scheduler.inbox.put(
            IncomingMessage(
                request_id=request_id,
                type="new_request",
                data=_stream_payload(request_id, **payload_kwargs),
            )
        )
    first = scheduler.inbox.get()
    scheduler._handle_message(first, None)
    assert flow.decoder.estimator.calls
    assert flow.decoder.estimator.calls[0]["streaming"] is True
    assert flow.decoder.estimator.calls[0]["x"].shape[0] == 4
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "stream"]


def test_c1_first_hop_does_not_wait_for_peers() -> None:
    flow, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler._ingest_stream_item("req-a", _item(list(range(28))))
    started = time.monotonic()
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert time.monotonic() - started < 0.05
    assert len(flow.calls) == 1


def test_queued_peer_chunk_joins_first_hop_batch_during_wait() -> None:
    from sglang_omni.models.fun_cosyvoice3.stages import FunCosyVoice3Flow
    from tests.unit_test.fun_cosyvoice3.test_flow_batch import _FakeFlow as _PackedFlow

    flow = _PackedFlow(channels=80, max_frames=128)
    flow.spk_embed_affine_layer = torch.nn.Linear(192, 80, bias=False)
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        stages.CosyVoice3Vocoder(FunCosyVoice3Flow(flow), _FakeHiFT()),
        max_batch_size=8,
    )
    prompt_len = TOKEN_HOP_LEN
    payload_kwargs = {
        "prompt_token_len": prompt_len,
        "prompt_feat_frames": prompt_len * TOKEN_MEL_RATIO,
    }
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(
            request_id, _stream_payload(request_id, **payload_kwargs)
        )
    scheduler._ingest_stream_item("req-a", _item(list(range(28))))
    scheduler.inbox.put(
        IncomingMessage(
            request_id="req-b",
            type="stream_chunk",
            data=_item(list(range(28))),
        )
    )
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert flow.decoder.estimator.calls[0]["x"].shape[0] == 4


def _packed_scheduler(*, max_batch_size: int = 8):
    from sglang_omni.models.fun_cosyvoice3.stages import FunCosyVoice3Flow
    from tests.unit_test.fun_cosyvoice3.test_flow_batch import _FakeFlow as _PackedFlow

    flow = _PackedFlow(channels=80, max_frames=512)
    flow.spk_embed_affine_layer = torch.nn.Linear(192, 80, bias=False)
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        stages.CosyVoice3Vocoder(FunCosyVoice3Flow(flow), _FakeHiFT()),
        max_batch_size=max_batch_size,
    )
    return flow, scheduler


def _aligned_payload(request_id: str) -> StagePayload:
    prompt_len = TOKEN_HOP_LEN
    return _stream_payload(
        request_id,
        prompt_token_len=prompt_len,
        prompt_feat_frames=prompt_len * TOKEN_MEL_RATIO,
    )


def test_equal_follow_up_hops_share_one_causal_flow_batch() -> None:
    flow, scheduler = _packed_scheduler()
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(request_id, _aligned_payload(request_id))
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    first_calls = len(flow.decoder.estimator.calls)
    assert flow.decoder.estimator.calls[0]["x"].shape[0] == 4

    for request_id in ("req-a", "req-b"):
        scheduler._ingest_stream_item(
            request_id, _item([i % 31 for i in range(28, 78)], chunk_id=1)
        )
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    follow_calls = flow.decoder.estimator.calls[first_calls:]
    assert follow_calls
    assert follow_calls[0]["streaming"] is True
    assert follow_calls[0]["x"].shape[0] == 4
    messages = [m for m in _drain(scheduler) if m.type == "stream"]
    shapes = {_waveform(m.data).shape[0] for m in messages}
    assert 100 in shapes


def test_mixed_prompt_follow_ups_share_one_causal_flow_batch() -> None:
    flow, scheduler = _packed_scheduler()
    payloads = {
        "req-a": _stream_payload(
            "req-a",
            prompt_token_len=TOKEN_HOP_LEN,
            prompt_feat_frames=TOKEN_HOP_LEN * TOKEN_MEL_RATIO,
        ),
        "req-b": _stream_payload(
            "req-b",
            prompt_token_len=TOKEN_HOP_LEN * 2,
            prompt_feat_frames=TOKEN_HOP_LEN * 2 * TOKEN_MEL_RATIO,
        ),
    }
    for request_id, payload in payloads.items():
        scheduler._on_streaming_new_request(request_id, payload)
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    first_calls = len(flow.decoder.estimator.calls)
    assert flow.decoder.estimator.calls[0]["x"].shape[0] == 4

    for request_id in payloads:
        scheduler._ingest_stream_item(
            request_id, _item([i % 31 for i in range(28, 78)], chunk_id=1)
        )
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    follow_calls = flow.decoder.estimator.calls[first_calls:]
    assert follow_calls
    assert follow_calls[0]["streaming"] is True
    assert follow_calls[0]["x"].shape[0] == 4


def test_c1_follow_up_stays_native_and_does_not_wait() -> None:
    flow, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler._ingest_stream_item("req-a", _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    scheduler._ingest_stream_item("req-a", _item(list(range(28, 78)), chunk_id=1))
    started = time.monotonic()
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert time.monotonic() - started < 0.05
    assert len(flow.calls) == 2
    assert int(flow.calls[1]["token"].shape[1]) == 78


def test_queued_peer_chunk_joins_follow_up_batch_during_wait() -> None:
    flow, scheduler = _packed_scheduler()
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(request_id, _aligned_payload(request_id))
        scheduler._ingest_stream_item(request_id, _item(list(range(28))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    first_calls = len(flow.decoder.estimator.calls)

    scheduler._ingest_stream_item(
        "req-a", _item([i % 31 for i in range(28, 78)], chunk_id=1)
    )
    scheduler.inbox.put(
        IncomingMessage(
            request_id="req-b",
            type="stream_chunk",
            data=_item([i % 31 for i in range(28, 78)], chunk_id=1),
        )
    )
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    follow_calls = flow.decoder.estimator.calls[first_calls:]
    assert follow_calls[0]["x"].shape[0] == 4


def test_backlogged_request_runs_one_hop_per_step() -> None:
    # note (guozhihao-224): 178 tokens cover first hop + two follow-ups
    # (28 / 78 / 178 windows). One step must advance only one hop.
    flow, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler._ingest_stream_item("req-a", _item(list(range(178))))
    with scheduler._state_lock:
        assert scheduler._pump_one_step() is None
    assert len(flow.calls) == 1
    assert int(flow.calls[0]["token"].shape[1]) == 28
    state = scheduler._stream_states["req-a"]
    assert state.token_offset == TOKEN_HOP_LEN
    assert state.hop_len == next_stream_hop_len(TOKEN_HOP_LEN)

    with scheduler._state_lock:
        assert scheduler._pump_one_step() is None
    assert len(flow.calls) == 2
    assert int(flow.calls[1]["token"].shape[1]) == 78
    assert state.token_offset == TOKEN_HOP_LEN + next_stream_hop_len(TOKEN_HOP_LEN)


def test_pump_drains_backlog_across_one_hop_steps() -> None:
    flow, scheduler = _scheduler()
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    scheduler._ingest_stream_item("req-a", _item(list(range(178))))
    with scheduler._state_lock:
        failed = scheduler._pump_streams()
    assert failed == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [28, 78, 178]
    messages = [m for m in _drain(scheduler) if m.type == "stream"]
    assert len(messages) == 3


def test_inbox_first_hop_preempts_follow_up_backlog() -> None:
    # note (guozhihao-224): after A's first hop, leave two follow-ups ready
    # and park B's first hop in the inbox. Between steps the pump must
    # ingest B and prefer that first hop over draining A's backlog.
    flow, scheduler = _scheduler()
    for request_id in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(request_id, _stream_payload(request_id))
    scheduler._ingest_stream_item("req-a", _item(list(range(28))))
    with scheduler._state_lock:
        assert scheduler._pump_streams() == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [28]

    scheduler._ingest_stream_item("req-a", _item(list(range(28, 178)), chunk_id=1))
    scheduler.inbox.put(
        IncomingMessage(
            request_id="req-b",
            type="stream_chunk",
            data=_item(list(range(28))),
        )
    )
    with scheduler._state_lock:
        assert scheduler._pump_streams() == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [
        28,
        78,
        28,
        178,
    ]
    messages = [m for m in _drain(scheduler) if m.type == "stream"]
    assert len(messages) == 4


def test_out_of_order_chunk_arrival_is_ingested_in_chunk_id_order() -> None:
    # Ordering is per request, not global: a chunk that arrives ahead of an
    # older chunk of the same request is held and ingested later, in chunk_id
    # order, so no chunk is dropped and none is appended out of order.
    flow, scheduler = _scheduler(max_batch_size=8)
    tokens = list(range(53))
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a", codes=tokens))

    scheduler._handle_message(_chunk_message("req-a", tokens[28:], chunk_id=1), None)

    state = scheduler._stream_states["req-a"]
    assert state.tokens == []
    assert _drain(scheduler) == []
    assert flow.calls == []

    scheduler._handle_message(_chunk_message("req-a", tokens[:28], chunk_id=0), None)

    # The missing head is ingested first; the first hop decodes from that prefix.
    assert state.tokens[:28] == tokens[:28]
    assert [int(call["token"].shape[1]) for call in flow.calls] == [28]

    scheduler._handle_message(_done_message("req-a"), None)
    _run_serving_loop(scheduler)

    messages = _drain(scheduler)
    assert [message.type for message in messages].count("result") == 1
    assert messages[-1].type == "result"
    assert flow.calls[-1]["finalize"] is True
    assert flow.calls[-1]["token"].flatten().tolist() == tokens
    audio = np.concatenate(
        [_waveform(message.data) for message in messages if message.type == "stream"]
    )
    np.testing.assert_array_equal(audio, np.arange(len(tokens) * TOKEN_MEL_RATIO))


def test_replayed_chunk_id_is_ignored() -> None:
    # A chunk at or below the request's next expected chunk_id re-delivers tokens
    # that are already ingested; appending it again would duplicate the audio.
    flow, scheduler = _scheduler(max_batch_size=8)
    tokens = list(range(53))
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a", codes=tokens))

    scheduler._handle_message(_chunk_message("req-a", tokens[:28], chunk_id=0), None)
    assert scheduler._stream_states["req-a"].tokens == tokens[:28]

    scheduler._handle_message(_chunk_message("req-a", tokens[:28], chunk_id=0), None)
    assert scheduler._stream_states["req-a"].tokens == tokens[:28]

    scheduler._handle_message(_chunk_message("req-a", tokens[28:], chunk_id=1), None)
    scheduler._handle_message(_done_message("req-a"), None)
    _run_serving_loop(scheduler)

    messages = _drain(scheduler)
    assert flow.calls[-1]["token"].flatten().tolist() == tokens
    audio = np.concatenate(
        [_waveform(message.data) for message in messages if message.type == "stream"]
    )
    np.testing.assert_array_equal(audio, np.arange(len(tokens) * TOKEN_MEL_RATIO))


def test_stream_done_consumes_a_parked_chunk_before_completing() -> None:
    # The collector parks a chunk it cannot batch, so a done can be dequeued
    # while an older chunk of the same request is still in `_pending_messages`.
    # Completing there clears the state and silently drops that chunk, and
    # waiting for the serving loop to walk the parked queue can outlive any
    # sane bound while a long pump step runs.
    flow, scheduler = _scheduler(max_batch_size=8)
    tokens = list(range(53))
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a", codes=tokens))
    scheduler._handle_message(_chunk_message("req-a", tokens[:28], chunk_id=0), None)
    first_hop = _drain(scheduler)
    assert [message.type for message in first_hop] == ["stream"]

    parked = _chunk_message("req-a", tokens[28:], chunk_id=1)
    scheduler._pending_messages.append(parked)

    scheduler._handle_message(_done_message("req-a"), None)

    # The done takes the parked chunk with it rather than finalizing without it.
    assert not scheduler._pending_messages
    _run_serving_loop(scheduler)

    messages = _drain(scheduler)
    assert [message.type for message in messages].count("result") == 1
    assert messages[-1].type == "result"
    assert flow.calls[-1]["finalize"] is True
    assert flow.calls[-1]["token"].flatten().tolist() == tokens
    audio = np.concatenate(
        [
            _waveform(message.data)
            for message in [*first_hop, *messages]
            if message.type == "stream"
        ]
    )
    np.testing.assert_array_equal(audio, np.arange(len(tokens) * TOKEN_MEL_RATIO))
    assert "req-a" not in scheduler._stream_states
    # A completed request never recreates state, so its late chunks stay dropped.
    assert scheduler._get_or_create_stream_state("req-a") is None


def test_stream_done_with_a_contract_violating_parked_chunk_owns_the_failure() -> None:
    # Twin of the parked-chunk test above, with a chunk that violates the
    # stream-chunk contract instead of one to ingest: a chunk that arrives ahead
    # of its predecessor is held (so the done waits on it), the collector parks
    # a second chunk of the same request, and the delivery that follows the
    # parked one resumes the done -- whose drain then meets that parked chunk.
    # The done path owns the failure: one terminal error (never a second from
    # the resume that dispatched the done, never a result), exactly one external
    # abort cleanup and never while ``_state_lock`` is held, no done or reorder
    # residue, and a serving loop that goes on serving the next request.
    _, scheduler = _scheduler(max_batch_size=8)
    cleanup_calls: list[str] = []
    cleanup_saw_lock_held: list[bool] = []

    def abort_callback(request_id: str) -> None:
        cleanup_saw_lock_held.append(scheduler._state_lock._is_owned())
        cleanup_calls.append(request_id)

    # This model scheduler is built without an external abort callback, so
    # install the one the pipeline would own.
    scheduler._abort_callback = abort_callback
    scheduler.inbox.put(
        IncomingMessage("req-bad", "new_request", _stream_payload("req-bad"))
    )
    scheduler.inbox.put(_chunk_message("req-bad", list(range(5)), chunk_id=1))
    scheduler.inbox.put(_done_message("req-bad"))
    scheduler.inbox.put(_chunk_message("req-bad", list(range(5, 10)), chunk_id=0))
    scheduler.inbox.put(IncomingMessage("req-bad", "stream_chunk", "not-a-stream-item"))
    scheduler.inbox.put(
        IncomingMessage("req-next", "new_request", _stream_payload("req-next"))
    )
    scheduler.inbox.put(_chunk_message("req-next", list(range(28)), chunk_id=0))
    scheduler.inbox.put(_done_message("req-next"))

    out = _serve(scheduler, output_count=4)

    assert [(message.type, message.request_id) for message in out] == [
        ("error", "req-bad"),
        ("stream", "req-next"),
        ("stream", "req-next"),
        ("result", "req-next"),
    ]
    # The failing request's one terminal message carries its contract error.
    assert isinstance(out[0].data, TypeError)
    assert cleanup_calls == ["req-bad"]
    assert cleanup_saw_lock_held == [False]
    assert "req-bad" not in scheduler._pending_done
    assert "req-bad" not in scheduler._deferred_done_deadlines
    assert not scheduler._deferred_chunks.get("req-bad")
    assert "req-bad" not in scheduler._stream_payloads
    assert "req-bad" not in scheduler._stream_states
    assert not scheduler._pending_messages


def test_stream_done_defers_while_a_chunk_waits_in_the_reorder_buffer() -> None:
    # A chunk whose predecessor is still in flight is held in the request's
    # reorder buffer, with nothing of that request left in the queues: completion
    # must still wait, or it finalizes a short utterance and drops the held tokens.
    flow, scheduler = _scheduler(max_batch_size=8)
    tokens = list(range(78))
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a", codes=tokens))

    scheduler._handle_message(_chunk_message("req-a", tokens[53:], chunk_id=2), None)
    assert scheduler._stream_states["req-a"].tokens == []
    assert not scheduler._pending_messages

    scheduler._handle_message(_done_message("req-a"), None)

    assert _drain(scheduler) == []
    assert scheduler._get_or_create_stream_state("req-a") is not None

    for chunk_id, codes in enumerate((tokens[:28], tokens[28:53])):
        scheduler._handle_message(
            _chunk_message("req-a", codes, chunk_id=chunk_id), None
        )
    _run_serving_loop(scheduler)

    messages = _drain(scheduler)
    assert [message.type for message in messages].count("result") == 1
    assert messages[-1].type == "result"
    assert flow.calls[-1]["finalize"] is True
    assert flow.calls[-1]["token"].flatten().tolist() == tokens
    audio = np.concatenate(
        [_waveform(message.data) for message in messages if message.type == "stream"]
    )
    np.testing.assert_array_equal(audio, np.arange(len(tokens) * TOKEN_MEL_RATIO))


def test_stream_done_wait_bound_completes_once_and_drops_the_held_chunk(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A chunk id that never arrives must not hang the request forever: once the
    # wait bound passes the done completes anyway, warning about the missing
    # chunk, and the held chunk is dropped without leaving bookkeeping behind.
    clock = _FakeClock()
    monkeypatch.setattr(streaming_simple_scheduler, "time", clock)
    _, scheduler = _scheduler(max_batch_size=8)
    tokens = list(range(78))
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a", codes=tokens))
    scheduler._handle_message(_chunk_message("req-a", tokens[53:], chunk_id=2), None)
    scheduler._handle_message(_done_message("req-a"), None)
    assert "req-a" in scheduler._deferred_done_deadlines
    assert "req-a" in scheduler._pending_done
    assert scheduler._deferred_chunks.get("req-a")

    clock.advance(_STREAM_DONE_MAX_WAIT_S + 1.0)
    with caplog.at_level(logging.WARNING, logger=streaming_simple_scheduler.__name__):
        assert _resume_without_raising(scheduler) == []

    assert [
        message.request_id for message in _drain(scheduler) if message.type == "result"
    ] == ["req-a"]
    assert any(
        record.levelno == logging.WARNING
        and "req-a" in record.getMessage()
        and "waited" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]
    assert "req-a" not in scheduler._deferred_done_deadlines
    assert "req-a" not in scheduler._pending_done
    assert not scheduler._deferred_chunks.get("req-a")
    assert "req-a" not in scheduler._stream_states

    # The late chunk cannot recreate the completed request, and no second result
    # is emitted for it.
    scheduler._handle_message(_chunk_message("req-a", tokens[53:], chunk_id=2), None)
    assert _resume_without_raising(scheduler) == []
    assert _drain(scheduler) == []
    assert "req-a" not in scheduler._stream_states
    assert scheduler._get_or_create_stream_state("req-a") is None


def test_failing_deferred_done_is_not_attributed_to_an_innocent_chunk_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A chunk delivery for one request resumes every deferred done. When another
    # request's resume raises, the failure must stay with that request: the
    # delivery must not surface it, and a healthy deferred done must still
    # complete.
    clock = _FakeClock()
    monkeypatch.setattr(streaming_simple_scheduler, "time", clock)
    _, scheduler = _scheduler(max_batch_size=8)
    tokens = list(range(78))
    # req-a never ingests a token, so its final decode has nothing to decode and
    # the fallback raises inside on_stream_done.
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a", codes=[]))
    scheduler._handle_message(_chunk_message("req-a", tokens[53:], chunk_id=2), None)
    scheduler._handle_message(_done_message("req-a"), None)
    # req-c is healthy and also waiting on a gap of its own.
    scheduler._on_streaming_new_request(
        "req-c", _stream_payload("req-c", codes=list(range(53)))
    )
    scheduler._handle_message(_chunk_message("req-c", tokens[53:], chunk_id=2), None)
    scheduler._handle_message(_done_message("req-c"), None)
    assert set(scheduler._deferred_done_deadlines) == {"req-a", "req-c"}

    clock.advance(_STREAM_DONE_MAX_WAIT_S + 1.0)
    assert (
        _resume_without_raising(scheduler) == []
    ), "a failing deferred done escaped the resume instead of erroring its own request"

    assert sorted(
        (message.type, message.request_id) for message in _drain(scheduler)
    ) == [
        ("error", "req-a"),
        ("result", "req-c"),
        ("stream", "req-c"),
    ]
    assert scheduler._is_aborted("req-a")
    assert not scheduler._is_aborted("req-c")
    assert not scheduler._deferred_done_deadlines
    assert not scheduler._pending_done
    assert not scheduler._deferred_chunks.get("req-a")
    assert not scheduler._deferred_chunks.get("req-c")


def test_backlogged_chunks_stay_ordered_before_stream_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow, scheduler = _scheduler(max_batch_size=8)
    tokens = list(range(78))
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a", codes=tokens))
    for chunk_id, codes in enumerate((tokens[:28], tokens[28:53])):
        scheduler.inbox.put(_chunk_message("req-a", codes, chunk_id=chunk_id))
    original_inference = flow.inference

    def inference(**kwargs):
        result = original_inference(**kwargs)
        if len(flow.calls) == 1:
            # The AR producer adds a newer chunk and completion while Flow runs
            # the first hop, so the pump reads that inbox between steps.
            scheduler.inbox.put(_chunk_message("req-a", tokens[53:], chunk_id=2))
            scheduler.inbox.put(_done_message("req-a"))
        return result

    monkeypatch.setattr(flow, "inference", inference)

    _run_serving_loop(scheduler)

    # The collector parks chunk 1 (its request is already in the batch), the pump
    # ingests chunk 2 ahead of it and the done arrives before chunk 1 is served.
    # Per-request ordering must still finalize with every chunk, exactly once.
    assert flow.calls[-1]["finalize"] is True
    assert flow.calls[-1]["token"].flatten().tolist() == tokens
    messages = _drain(scheduler)
    assert [m.type for m in messages].count("result") == 1
    assert messages[-1].type == "result"
    audio = np.concatenate([_waveform(m.data) for m in messages if m.type == "stream"])
    np.testing.assert_array_equal(audio, np.arange(len(tokens) * TOKEN_MEL_RATIO))
    assert "req-a" not in scheduler._stream_states


@pytest.mark.parametrize("streaming", [False, True])
def test_new_request_collection_never_batches_a_parked_chunk(streaming: bool) -> None:
    # Only terminal payloads batch; a chunk parked ahead of a payload stays
    # queued for the serving loop and is never consumed twice or lost.
    _, scheduler = _scheduler(max_batch_size=8)
    payload = _stream_payload("req-a")
    payload.request.params["stream"] = streaming
    first = IncomingMessage(request_id="req-a", type="new_request", data=payload)
    parked = _chunk_message("req-a", [2], chunk_id=0)
    newer = _chunk_message("req-a", [3], chunk_id=1)
    scheduler._pending_messages.append(parked)
    scheduler.inbox.put(newer)

    batch = scheduler._collect_new_request_batch(first)

    assert batch == [first]
    assert sorted(m.data.chunk_id for m in scheduler._pending_messages) == [0, 1]
    assert scheduler.inbox.empty()


def test_chunk_collection_batches_inbox_peers_and_keeps_parked_chunks() -> None:
    _, scheduler = _scheduler(max_batch_size=8)
    first = _chunk_message("a", [1], chunk_id=0)
    parked = _chunk_message("a", [2], chunk_id=1)
    peer_b = _chunk_message("b", [4], chunk_id=0)
    peer_c = _chunk_message("c", [5], chunk_id=0)
    done = _done_message("a")
    later = _chunk_message("d", [6], chunk_id=0)
    scheduler._pending_messages.append(parked)
    for message in (peer_b, peer_c, done, later):
        scheduler.inbox.put(message)

    batch = scheduler._collect_stream_chunk_batch(first)

    # Peers queued behind the first chunk coalesce into the batch; the done stops
    # the scan. Every other message is still queued exactly once.
    assert [message.request_id for message in batch] == ["a", "b", "c"]
    remaining = [*scheduler._pending_messages, *_drain_inbox(scheduler)]
    assert sorted(id(message) for message in [*batch, *remaining]) == sorted(
        id(message) for message in (first, parked, peer_b, peer_c, done, later)
    )


def test_streaming_payload_collection_reads_the_inbox_not_the_parked_deque() -> None:
    _, scheduler = _scheduler(max_batch_size=3)
    messages = [
        IncomingMessage(rid, "new_request", _stream_payload(rid))
        for rid in ("a", "b", "c", "d")
    ]
    parked = messages[1:3]
    scheduler._pending_messages.extend(parked)
    scheduler.inbox.put(messages[3])

    assert scheduler._collect_new_request_batch(messages[0]) == [
        messages[0],
        messages[3],
    ]
    assert list(scheduler._pending_messages) == parked
    assert scheduler.inbox.empty()


@pytest.mark.parametrize("coalescing", [False, True])
def test_non_streaming_batch_reads_the_inbox_past_a_parked_done_marker(
    coalescing: bool,
) -> None:
    # The cost cap still bounds the batch, a parked done marker does not stall
    # the scan, and the parked messages stay for `_next_message`.
    _, scheduler = _scheduler(
        max_batch_size=8, request_cost_fn=lambda payload: 1, max_batch_cost=2
    )
    scheduler._can_batch_stream_chunks = coalescing
    messages = []
    for rid in ("a", "b", "c", "d"):
        payload = _stream_payload(rid)
        payload.request.params["stream"] = False
        messages.append(IncomingMessage(rid, "new_request", payload))
    parked = [IncomingMessage("b", "stream_done"), messages[1], messages[2]]
    scheduler._pending_messages.extend(parked)
    scheduler.inbox.put(messages[3])

    assert scheduler._collect_new_request_batch(messages[0]) == [
        messages[0],
        messages[3],
    ]
    assert list(scheduler._pending_messages) == parked
    assert scheduler.inbox.empty()


def test_queued_peer_payloads_still_share_a_causal_flow_batch() -> None:
    flow, scheduler = _packed_scheduler()
    first = IncomingMessage("a", "new_request", _aligned_payload("a"))
    peer = IncomingMessage("b", "new_request", _aligned_payload("b"))
    for rid in ("a", "b"):
        scheduler._ingest_stream_item(rid, _item(list(range(28)), chunk_id=0))
    scheduler.inbox.put(peer)

    batch = scheduler._collect_new_request_batch(first)
    assert batch == [first, peer]
    scheduler._handle_new_request_batch(batch)

    assert flow.decoder.estimator.calls
    # CFG packs each request twice: two requests share one causal Flow call.
    assert flow.decoder.estimator.calls[0]["x"].shape[0] == 4
    assert len([msg for msg in _drain(scheduler) if msg.type == "stream"]) == 2


@pytest.mark.parametrize(
    "reader",
    ["_ingest_ready_inbox", "_wait_for_first_hop_peers", "_wait_for_follow_up_peers"],
)
def test_peer_readers_ingest_the_inbox_past_parked_messages(reader: str) -> None:
    # Ordering is request-local, not a global FIFO: a message parked for one
    # request must neither stall the reader nor be drained (or passed) by it.
    _, scheduler = _scheduler(max_batch_size=8)
    for rid in ("req-a", "req-b"):
        scheduler._on_streaming_new_request(rid, _stream_payload(rid))
    follow_up = reader == "_wait_for_follow_up_peers"
    target = 78 if follow_up else 28
    start = 28 if follow_up else 14
    scheduler._ingest_stream_item("req-a", _item(list(range(target)), chunk_id=0))
    scheduler._ingest_stream_item("req-b", _item(list(range(start)), chunk_id=0))
    if follow_up:
        for state in scheduler._stream_states.values():
            state.token_offset = 25
            state.hop_len = 50
    parked = _chunk_message("req-a", list(range(target, target + 25)), chunk_id=1)
    scheduler._pending_messages.append(parked)
    scheduler.inbox.put(_chunk_message("req-b", list(range(start, target)), chunk_id=1))

    getattr(scheduler, reader)()

    # Request B is served from the inbox while request A keeps its parked chunk.
    assert scheduler._stream_states["req-b"].tokens == list(range(target))
    assert any(message is parked for message in scheduler._pending_messages)
    assert scheduler.inbox.empty()


def test_disable_hop_growth_keeps_fixed_follow_up_windows() -> None:
    flow, scheduler = _scheduler(disable_hop_growth=True)
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    # First hop 28, then two fixed 25-token hops -> need 28+25+25 = 78 tokens
    # for three windows of 28 / 53 / 78 (lookahead included in prefix).
    scheduler._ingest_stream_item("req-a", _item(list(range(78))))
    with scheduler._state_lock:
        assert scheduler._pump_streams() == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [28, 53, 78]
    state = scheduler._stream_states["req-a"]
    assert state.hop_len == TOKEN_HOP_LEN


def test_token_max_hop_len_caps_growth() -> None:
    flow, scheduler = _scheduler(token_max_hop_len=50)
    scheduler._on_streaming_new_request("req-a", _stream_payload("req-a"))
    # With max 50: hops 25 -> 50 -> 50. Windows 28 / 78 / 128.
    scheduler._ingest_stream_item("req-a", _item(list(range(128))))
    with scheduler._state_lock:
        assert scheduler._pump_streams() == []
    assert [int(call["token"].shape[1]) for call in flow.calls] == [28, 78, 128]
    state = scheduler._stream_states["req-a"]
    assert state.hop_len == 50
