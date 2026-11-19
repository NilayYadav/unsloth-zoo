# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

if "mlx_simulation" in (getattr(mx, "__file__", "") or ""):
    pytest.skip("mlx.core is the tests/mlx_simulation stub, not real MLX",
                allow_module_level = True)

from mlx.utils import tree_flatten                                 # noqa: E402

from unsloth_zoo.mlx import utils as mlx_utils                     # noqa: E402


class _DropoutLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.5)

    def __call__(self, x):
        return x + self.proj(self.dropout(x))


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_DropoutLayer() for _ in range(3)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _grads(model, x):
    mx.random.seed(7)
    loss, grad = nn.value_and_grad(model, lambda: (model(x) ** 2).sum())()
    mx.eval(loss, grad)
    return loss.item(), dict(tree_flatten(grad))


def test_checkpointed_dropout_backward_uses_the_forward_mask(monkeypatch):
    mx.random.seed(0)
    model = _Stack()
    x = mx.random.normal((4, 32))
    mlx_utils._patch_layer_class_for_gc(_DropoutLayer)
    try:
        loss, grad = _grads(model, x)
        monkeypatch.setattr(mx, "checkpoint", lambda fn: fn)
        expected_loss, expected = _grads(model, x)
    finally:
        mlx_utils._unpatch_layer_class_gc(_DropoutLayer)

    assert loss == pytest.approx(expected_loss, rel = 1e-6)
    for name, value in expected.items():
        assert mx.allclose(grad[name], value, rtol = 1e-4, atol = 1e-5).item(), name
