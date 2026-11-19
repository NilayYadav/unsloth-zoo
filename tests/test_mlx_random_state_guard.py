from functools import partial

import pytest


def test_a_sampler_that_raises_inside_compile_leaves_later_draws_working():
    mx = pytest.importorskip("mlx.core")
    from mlx_simulation import mlx_is_simulated

    if mlx_is_simulated():
        pytest.skip("needs real mlx: the shim has no compiled random state")
    from unsloth_zoo.mlx.inference import inference_scopes

    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def refusing_sampler(logits):
        mx.random.categorical(logits)
        raise ValueError("top_k out of range")

    mx.random.seed(3)
    key = mx.random.state[0].tolist()
    with pytest.raises(ValueError, match="top_k"):
        with inference_scopes(object()):
            refusing_sampler(mx.zeros((1, 4)))
    mx.eval(mx.random.state)
    assert mx.random.state[0].tolist() == key
    assert mx.random.categorical(mx.zeros((1, 4))).shape == (1,)
