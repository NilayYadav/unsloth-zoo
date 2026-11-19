# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""An unformatted Alpaca row is not training text: CUDA asks for a formatting_func, MLX must not train on the instruction alone."""

import pytest

pytest.importorskip("mlx.core")

ALPACA_ROW = {
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "Eat well, sleep enough and exercise.",
}


class _Tokenizer:
    eos_token_id = 2
    chat_template = None

    def encode(self, text, add_special_tokens=True):
        return [3 + ord(c) % 50 for c in text]


def test_unformatted_alpaca_row_renders_no_text():
    from unsloth_zoo.mlx.utils import collect_mlx_texts

    assert collect_mlx_texts(_Tokenizer(), ALPACA_ROW) == []
    assert collect_mlx_texts(_Tokenizer(), {**ALPACA_ROW, "text": "formatted"}) == ["formatted"]


def test_unformatted_alpaca_dataset_is_refused():
    from unsloth_zoo.mlx.utils import _prepare_dataset

    with pytest.raises(ValueError, match="No text data found"):
        _prepare_dataset([ALPACA_ROW], _Tokenizer(), append_eos=False)
