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

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

if "mlx_simulation" in (getattr(mx, "__file__", "") or ""):
    pytest.skip("mlx.core is the tests/mlx_simulation stub, not real MLX",
                allow_module_level = True)

from types import SimpleNamespace                                  # noqa: E402

from mlx.utils import tree_flatten                                 # noqa: E402

from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig  # noqa: E402
from unsloth_zoo.mlx.utils import FiniteTextBatchPlan, _FiniteTextRow  # noqa: E402

VOCAB, DIM, ROWS, WIDTH = 64, 16, 8, 513


class _HalfLM(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(0)
        self.embed = nn.Embedding(VOCAB, DIM)
        self.adapter = nn.Linear(DIM, DIM, bias = False)
        self.head = nn.Linear(DIM, VOCAB, bias = False)
        self.head.weight = self.head.weight * 0.002
        self._config = {"model_type": "tiny"}

    def __call__(self, input_ids):
        hidden = self.embed(input_ids)
        hidden = hidden + self.adapter(hidden.astype(mx.float32)).astype(hidden.dtype)
        return self.head(hidden)

    def train(self, mode = True):
        return self


def _rows():
    rng = np.random.default_rng(0)
    return tuple(
        _FiniteTextRow(tuple(int(t) for t in rng.integers(1, VOCAB, WIDTH)), offset = 1, labels = None)
        for _ in range(ROWS)
    )


def _reference_grad(model, rows):
    ref = _HalfLM()
    ref.update(model.parameters())
    ref.set_dtype(mx.float32)
    ref.freeze()
    ref.adapter.unfreeze()
    batch = mx.array([row.input_ids for row in rows])

    def loss():
        logits = ref(batch[:, :-1])
        return nn.losses.cross_entropy(logits, batch[:, 1:], reduction = "mean")

    return dict(tree_flatten(nn.value_and_grad(ref, loss)()[1]))["adapter.weight"]


@pytest.mark.parametrize("grad_accum", [1, 2])
def test_float16_training_gradient_does_not_underflow(tmp_path, grad_accum):
    model = _HalfLM()
    model.embed.set_dtype(mx.float16)
    model.head.set_dtype(mx.float16)
    model.freeze()
    model.adapter.unfreeze()
    rows = _rows()
    expected = _reference_grad(model, rows)
    before = mx.array(model.adapter.weight)
    mx.eval(expected, before)

    trainer = MLXTrainer(
        model,
        SimpleNamespace(pad_token_id = 0, eos_token_id = 1),
        [],
        args = MLXTrainingConfig(
            max_steps = 1,
            per_device_train_batch_size = ROWS // grad_accum,
            gradient_accumulation_steps = grad_accum,
            optim = "sgd",
            learning_rate = 1.0,
            lr_scheduler_type = "constant",
            warmup_steps = 0,
            weight_decay = 0.0,
            use_cce = False,
            gradient_checkpointing = False,
            cast_norm_output_to_input_dtype = False,
            max_grad_norm = 0.0,
            max_grad_leaf_norm = 0.0,
            disable_memory_limits = True,
            output_dir = str(tmp_path),
        ),
    )
    per_batch = ROWS // grad_accum
    trainer._batches = FiniteTextBatchPlan(
        rows,
        tuple(tuple(range(i, i + per_batch)) for i in range(0, ROWS, per_batch)),
        max_seq_length = WIDTH,
        pad_id = 0,
    )
    trainer.save_model = lambda *_args, **_kwargs: None
    trainer.train()

    observed = before - model.adapter.weight
    error = mx.linalg.norm(observed - expected) / mx.linalg.norm(expected)
    assert error.item() < 0.02, error.item()
