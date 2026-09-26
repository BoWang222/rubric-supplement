"""Isolated vLLM generation worker for the RubricARROW hybrid scorer.

The worker intentionally returns token IDs, not branch probabilities.  Exact
Boolean branch probabilities are computed later from unprocessed Hugging Face
logits so temperature/top-p processors can never censor the losing branch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _finish_isolated_worker(engine: LLM, timeout_seconds: float = 15.0) -> None:
    """Bound vLLM shutdown and skip interpreter teardown after durable output.

    vLLM 0.8.5's V1 EngineCore can finish every prompt and then wait forever
    while closing its ZMQ resources.  This module is already an isolated
    subprocess, so once the output and metrics files are closed it is safer to
    give graceful shutdown a short window and then terminate the worker
    directly.  ``os._exit`` also avoids a second, implicit shutdown during
    Python interpreter teardown.
    """
    finished = threading.Event()

    def shutdown() -> None:
        try:
            engine.llm_engine.engine_core.shutdown()
        except BaseException as error:  # pragma: no cover - vLLM-runtime only
            print(json.dumps({
                "event": "vllm_shutdown_error",
                "error": repr(error),
            }, sort_keys=True), file=sys.stderr, flush=True)
        finally:
            finished.set()

    thread = threading.Thread(target=shutdown, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if not finished.is_set():
        print(json.dumps({
            "event": "vllm_shutdown_timeout",
            "timeout_seconds": timeout_seconds,
        }, sort_keys=True), file=sys.stderr, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    args = parser.parse_args()

    requests = _rows(args.requests)
    if not requests:
        raise ValueError("vLLM request file is empty")
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    token_prompts = []
    for request in requests:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": request["rendered_user_content"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_prompts.append({"prompt_token_ids": list(map(int, prompt_ids))})

    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
        enable_prefix_caching=True,
    )
    initialized = time.monotonic()
    # A per-request SamplingParams object preserves the registered seed=42
    # contract independently of scheduler batch order.
    sampling = [
        SamplingParams(
            temperature=1.0,
            top_p=0.95,
            seed=args.seed,
            max_tokens=args.max_new_tokens,
        )
        for _ in requests
    ]
    outputs = engine.generate(token_prompts, sampling, use_tqdm=True)
    generated = time.monotonic()
    if len(outputs) != len(requests):
        raise AssertionError("vLLM changed request cardinality")

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for request, output in zip(requests, outputs, strict=True):
            choice = output.outputs[0]
            handle.write(json.dumps({
                "request_id": request["request_id"],
                "prompt_token_ids": list(map(int, output.prompt_token_ids)),
                "generated_token_ids": list(map(int, choice.token_ids)),
                "text": choice.text.strip(),
                "finish_reason": choice.finish_reason,
            }, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)
    args.metrics.write_text(json.dumps({
        "request_count": len(requests),
        "prompt_tokens": sum(len(output.prompt_token_ids) for output in outputs),
        "generated_tokens": sum(len(output.outputs[0].token_ids) for output in outputs),
        "engine_initialization_seconds": initialized - started,
        "batch_generation_seconds": generated - initialized,
        "total_seconds": generated - started,
    }, indent=2, sort_keys=True) + "\n")
    _finish_isolated_worker(engine)


if __name__ == "__main__":
    main()
