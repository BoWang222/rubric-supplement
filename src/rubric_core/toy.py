"""Create a tiny random local model to check training without downloads."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch


def build_model_and_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    words = ["<pad>", "<unk>", "<eos>", "user", "assistant", ":", "What", "is", "two", "plus",
             "?", "four", "five", "Name", "a", "color", "blue", "stone", "Explain", "briefly", "."]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                        pad_token="<pad>", eos_token="<eos>")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] }}{% if m['role'] == 'assistant' %}{{ eos_token }}{% endif %}{{ ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        bos_token_id=2, eos_token_id=2, pad_token_id=0, attention_dropout=0.0))
    return model, tokenizer


def main():
    from .io import read_jsonl
    from .train import run_training
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="examples/pairs.jsonl")
    parser.add_argument("--output", default="outputs/toy-training")
    parser.add_argument("--backend", choices=["none", "kl", "wasserstein_w1"], default="kl")
    args = parser.parse_args()
    model, tokenizer = build_model_and_tokenizer()
    run_training(model, tokenizer, read_jsonl(args.input), output=Path(args.output),
                 backend=args.backend, rho=.02, epochs=2, batch_size=1, lr=.001, max_steps=3)


if __name__ == "__main__":
    main()
