"""Minimal single-device MMPO training using cached reference log probabilities.

The loss and shared-dual rule are extracted from the experiment. This small
driver omits the production FSDP launcher and dataset-specific preprocessing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .io import read_jsonl, write_jsonl
from .objective import MMPOObjective


def completion_ids(tokenizer, messages, response, max_prompt=1024, max_completion=1024):
    if any(token in response for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
        raise ValueError("Response contains a reserved conversation control token")
    prompt = tokenizer.apply_chat_template(messages, tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": response}],
                                        tokenize=True, add_generation_prompt=False, enable_thinking=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("The tokenizer's assistant completion does not preserve the prompt prefix")
    completion = full[len(prompt):]
    eos = tokenizer.eos_token_id
    if eos is None or not prompt:
        raise ValueError("Tokenizer must provide EOS and a nonempty prompt")
    if eos in completion:
        index = completion.index(eos)
        if tokenizer.decode(completion[index + 1:]).strip():
            raise ValueError("Non-whitespace tokens follow assistant EOS")
        completion = completion[:index + 1]
    else:
        completion = completion + [eos]
    if len(completion) > max_completion:
        completion = completion[:max_completion - 1] + [eos]
    return list(prompt[-max_prompt:]), list(completion)


def summed_completion_logps(model, sequences, pad_id):
    """Each sequence is (prompt IDs, completion IDs); EOS is included."""
    device = next(model.parameters()).device
    width = max(len(p) + len(c) for p, c in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (prompt, completion) in enumerate(sequences):
        n = len(prompt) + len(completion)
        ids[i, :n] = torch.tensor(prompt + completion, device=device)
        attention[i, :n] = 1
        labels[i, len(prompt):n] = torch.tensor(completion, device=device)
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1]
    targets = labels[:, 1:]
    token_losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                  ignore_index=-100, reduction="none").reshape(targets.shape)
    return -token_losses.sum(-1)


def run_training(model, tokenizer, rows, *, output, backend="none", rho=0.0,
                 s_mode="context", epochs=1, batch_size=1, accumulation_steps=1,
                 lr=1e-6, beta=.01, gamma=2.2, seed=42, max_steps=None,
                 max_prompt=1024, max_completion=1024):
    if not rows or min(epochs, batch_size, accumulation_steps, max_prompt, max_completion) < 1:
        raise ValueError("Empty data or nonpositive training setting")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for key in ("attention_dropout", "hidden_dropout", "resid_pdrop", "embd_pdrop", "attn_pdrop"):
        if hasattr(model.config, key):
            setattr(model.config, key, 0.0)
    model.eval()
    prepared = []
    # The initial model is the frozen reference for this standalone run.
    # Cache first, then optimize the same model; no second 8B copy is retained.
    for row in rows:
        chosen = completion_ids(tokenizer, row["prompt_messages"], row["chosen"], max_prompt, max_completion)
        rejected = completion_ids(tokenizer, row["prompt_messages"], row["rejected"], max_prompt, max_completion)
        with torch.no_grad():
            reference = summed_completion_logps(model, [chosen, rejected], tokenizer.pad_token_id).cpu().tolist()
        margin, scale = float(row["margin_normalized"]), float(row["rubric_s"])
        if not 0 < margin <= 1 or not .05 <= scale <= 1:
            raise ValueError("margin_normalized or rubric_s is outside its valid range")
        prepared.append({"context_id": row["context_id"], "chosen": chosen, "rejected": rejected,
                         "reference": reference, "margin_normalized": margin, "rubric_s": scale})
    write_jsonl(output / "reference_cache.jsonl", prepared)
    device = next(model.parameters()).device
    objective = MMPOObjective(beta=beta, gamma=gamma, backend=backend, rho=rho, s_mode=s_mode, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(math.ceil(len(prepared) / batch_size) / accumulation_steps)
    total_steps = min(epochs * steps_per_epoch, max_steps or epochs * steps_per_epoch)
    warmup = math.ceil(.3 * total_steps)

    def schedule(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return .5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    generator = torch.Generator().manual_seed(seed)
    history, step = [], 0
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(prepared), generator=generator).tolist()
        micro_batches = [order[j:j + batch_size] for j in range(0, len(order), batch_size)]
        for start in range(0, len(micro_batches), accumulation_steps):
            group = micro_batches[start:start + accumulation_steps]
            optimizer.zero_grad(set_to_none=True)
            loss_sum, diagnostics = 0.0, {}
            group_count = sum(len(indices) for indices in group)
            for indices in group:
                items = [prepared[j] for j in indices]
                sequences = [item["chosen"] for item in items] + [item["rejected"] for item in items]
                logps = summed_completion_logps(model, sequences, tokenizer.pad_token_id)
                reference = torch.tensor([item["reference"] for item in items], device=device)
                margin = torch.tensor([item["margin_normalized"] for item in items], device=device)
                scale = torch.tensor([item["rubric_s"] for item in items], device=device)
                loss, diagnostics = objective(logps[:len(items)], logps[len(items):],
                                              reference[:, 0], reference[:, 1], margin, scale)
                weight = len(items) / group_count
                (loss * weight).backward()
                loss_sum += float(loss.detach()) * weight
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            record = {"step": step, "epoch": epoch, "loss": loss_sum, **diagnostics}
            history.append(record)
            print(json.dumps(record), flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    model.save_pretrained(output / "model", safe_serialization=True)
    tokenizer.save_pretrained(output / "model")
    write_jsonl(output / "training.jsonl", history)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "dual": objective.controller.checkpoint_state() if objective.controller else None},
               output / "training_state.pt")
    (output / "run.json").write_text(json.dumps({"rows": len(rows), "steps": step, "seed": seed,
        "backend": backend, "rho": rho, "s_mode": s_mode, "beta": beta, "gamma": gamma,
        "batch_size": batch_size, "accumulation_steps": accumulation_steps,
        "learning_rate": lr, "driver": "standalone single-device core demonstration"}, indent=2) + "\n")
    return history


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local initial-policy/reference checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--scores", help="Measured s JSONL; unmatched/failed contexts are excluded for every backend")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["none", "kl", "wasserstein_w1"], default="none")
    parser.add_argument("--s-mode", choices=["context", "uniform"], default="context")
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=.01)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--attention", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    if args.scores:
        scores = {row["context_id"]: row for row in read_jsonl(args.scores, unique_contexts=True) if row.get("status") == "complete"}
        retained = [{**row, "rubric_s": scores[row["context_id"]]["s"]} for row in rows if row["context_id"] in scores]
        print(json.dumps({"input_rows": len(rows), "retained_rows": len(retained), "excluded_rows": len(rows) - len(retained)}))
        rows = retained
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        attn_implementation=args.attention).to(args.device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    run_training(model, tokenizer, rows, output=args.output, backend=args.backend, rho=args.rho,
                 s_mode=args.s_mode, epochs=args.epochs, batch_size=args.batch_size,
                 accumulation_steps=args.accumulation_steps, lr=args.lr, beta=args.beta,
                 gamma=args.gamma, seed=args.seed, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
