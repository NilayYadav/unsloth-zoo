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

"""MLX SFT appends EOS to plain text like CUDA SFT: never after text that already ends in a special token."""

import pytest

pytest.importorskip("mlx.core")

EOS = 99
CHAT_ROW = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n"


class _Tokenizer:
    eos_token = "</s>"
    eos_token_id = EOS
    bos_token = None
    pad_token_id = 0
    chat_template = None
    all_special_tokens = ["</s>", "<|im_start|>", "<|im_end|>"]

    def encode(self, text, add_special_tokens=True):
        ids = [3 + ord(c) % 50 for c in text.removesuffix("</s>")]
        return ids + [EOS] if text.endswith("</s>") else ids


def _ends_with_eos(ids):
    return list(ids)[-1] == EOS and list(ids)[-2] != EOS


def test_prepare_dataset_eos_rule():
    from unsloth_zoo.mlx.utils import _prepare_dataset

    rows = [{"text": "plain text"}, {"text": "already ends</s>"}, {"text": CHAT_ROW}]
    dataset = _prepare_dataset(rows, _Tokenizer(), append_eos=True)
    ids = [dataset[i][0] for i in range(3)]
    assert _ends_with_eos(ids[0])
    assert _ends_with_eos(ids[1])
    assert ids[2] == _Tokenizer().encode(CHAT_ROW)


def test_labeled_batches_eos_rule():
    from unsloth_zoo.mlx.trainer import _create_labeled_batches

    rows = [{"text": "plain text"}, {"text": CHAT_ROW}]
    _, dataset = _create_labeled_batches(
        rows, _Tokenizer(), lambda batch: {"labels": batch["input_ids"]},
        batch_size=1, max_seq_length=512, dataset_order="sequential",
        append_eos=True, return_dataset=True,
    )
    by_length = sorted((row["input_ids"] for row in dataset), key=len)
    assert _ends_with_eos(by_length[0])
    assert by_length[1] == _Tokenizer().encode(CHAT_ROW)


def test_prompt_completion_string_rows_end_in_eos():
    from unsloth_zoo.mlx.utils import _tokenize_mlx_prompt_completion_row

    ids, labels = _tokenize_mlx_prompt_completion_row(
        _Tokenizer(), {"prompt": "The capital of France is", "completion": " Paris."},
        append_eos=True,
    )
    assert _ends_with_eos(ids) and labels[-1] == EOS
    ids, _ = _tokenize_mlx_prompt_completion_row(
        _Tokenizer(), {"prompt": "<|im_start|>user\nhi<|im_end|>\n", "completion": "<|im_start|>assistant\nhey<|im_end|>\n"},
        append_eos=True,
    )
    assert list(ids)[-1] != EOS
