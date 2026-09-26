import pytest
import torch

pytest.importorskip("transformers")
from rubric_core.toy import build_model_and_tokenizer
from rubric_core.train import run_training, summed_completion_logps


@pytest.mark.parametrize("backend", ["none", "kl", "wasserstein_w1"])
def test_small_model_updates_and_saves(tmp_path, backend):
    model, tokenizer = build_model_and_tokenizer()
    before = next(model.parameters()).detach().clone()
    rows = [{"context_id": "toy", "prompt_messages": [{"role": "user", "content": "What is two plus two ?"}],
             "chosen": "four", "rejected": "five", "margin_normalized": .8, "rubric_s": .3}]
    history = run_training(model, tokenizer, rows, output=tmp_path / backend, backend=backend,
                           rho=.02, lr=.001, epochs=2, batch_size=1, max_steps=2)
    assert len(history) == 2
    assert not torch.equal(before, next(model.parameters()).detach())
    assert (tmp_path / backend / "model/config.json").is_file()
    assert (tmp_path / backend / "training_state.pt").is_file()


def test_completion_mask_excludes_prompt_and_padding():
    model, tokenizer = build_model_and_tokenizer()
    model.eval()
    pairs = [([3, 5, 6, 7], [11, 2]), ([3, 5], [12, 2])]
    batch = summed_completion_logps(model, pairs, tokenizer.pad_token_id)
    single = torch.cat([summed_completion_logps(model, [pair], tokenizer.pad_token_id) for pair in pairs])
    torch.testing.assert_close(batch, single)
    for index, (prompt, completion) in enumerate(pairs):
        full = torch.tensor([prompt + completion])
        logits = model(input_ids=full).logits[0]
        expected = sum(torch.log_softmax(logits[len(prompt) - 1 + j].float(), -1)[token]
                       for j, token in enumerate(completion))
        torch.testing.assert_close(batch[index], expected)
