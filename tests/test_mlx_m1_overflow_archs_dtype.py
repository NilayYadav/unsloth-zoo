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

import json

import pytest

mx = pytest.importorskip("mlx.core")

if "mlx_simulation" in (getattr(mx, "__file__", "") or ""):
    pytest.skip("mlx.core is the tests/mlx_simulation stub, not real MLX",
                allow_module_level = True)

from mlx.utils import tree_flatten  # noqa: E402

_CONFIGS = {
    "gemma3_text": {
        "model_type": "gemma3_text", "hidden_size": 32, "num_hidden_layers": 2,
        "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 16, "vocab_size": 64, "rms_norm_eps": 1e-6, "sliding_window": 16,
        "rope_theta": 10000.0, "rope_local_base_freq": 10000.0,
        "query_pre_attn_scalar": 16, "max_position_embeddings": 128,
    },
    "llama": {
        "model_type": "llama", "hidden_size": 32, "num_hidden_layers": 2,
        "intermediate_size": 64, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 64, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "max_position_embeddings": 128, "tie_word_embeddings": True,
    },
}


def _checkpoint(tmp_path, model_type):
    import importlib

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    config = {**_CONFIGS[model_type], "torch_dtype": "bfloat16"}
    module = importlib.import_module(f"mlx_lm.models.{model_type}")
    model = module.Model(module.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (tmp_path / "config.json").write_text(json.dumps(config))
    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(64)}, unk_token = "t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tokenizer, unk_token = "t0", eos_token = "t1", pad_token = "t0",
    ).save_pretrained(str(tmp_path))
    return str(tmp_path)


@pytest.mark.parametrize("model_type, expected", [("gemma3_text", "bfloat16"), ("llama", "float16")])
def test_m1_default_dtype_keeps_overflowing_archs_out_of_float16(tmp_path, monkeypatch, model_type, expected):
    from unsloth_zoo.mlx.loader import FastMLXModel

    path = _checkpoint(tmp_path, model_type)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M1", "architecture": "applegpu_g13g"})
    model, _ = FastMLXModel.from_pretrained(path, load_in_4bit = False)

    dtypes = {v.dtype for _, v in tree_flatten(model.parameters()) if mx.issubdtype(v.dtype, mx.floating)}
    assert dtypes == {getattr(mx, expected)}, dtypes
