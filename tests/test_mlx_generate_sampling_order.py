import types

import pytest


def test_generate_applies_temperature_before_min_p_like_transformers(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from mlx_simulation import mlx_is_simulated

    if mlx_is_simulated():
        pytest.skip("needs real mlx sampling")
    import mlx_lm
    import unsloth_zoo.mlx.loader as loader

    logits = mx.array([[2.0, 1.0, 0.0, -1.0]])
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    drawn = set()

    def fake_stream_generate(_model, _tokenizer, prompt, max_tokens=None, sampler=None, **kwargs):
        mx.random.seed(0)
        drawn.update(int(sampler(logprobs).item()) for _ in range(300))
        yield types.SimpleNamespace(token=2)

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    model = types.SimpleNamespace(_tokenizer=types.SimpleNamespace(bos_token=None, eos_token_ids={2}), _is_vlm_model=False)
    loader._mlx_generate(model, input_ids=[[1, 3]], max_new_tokens=1, do_sample=True, temperature=2.0, min_p=0.3)
    # At temperature 2 the probabilities relative to the top token are 1, 0.61, 0.37, 0.22: min_p=0.3 keeps three.
    assert drawn == {0, 1, 2}
