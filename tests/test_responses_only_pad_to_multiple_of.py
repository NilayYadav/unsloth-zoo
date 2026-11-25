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

import inspect

import pytest

torch = pytest.importorskip("torch")
sft_trainer = pytest.importorskip("trl.trainer.sft_trainer")
from transformers import DataCollatorForSeq2Seq  # noqa: E402
from datasets import Dataset  # noqa: E402

from unsloth_zoo.dataset_utils import train_on_responses_only  # noqa: E402

from test_padding_free_collator_survives import (  # noqa: E402
    INSTRUCTION_PART, LONG_ROW, PAD_ID, RESPONSE_PART, SHORT_ROW, StubTokenizer, StubTrainer,
)


class PaddingTokenizer(StubTokenizer):
    def pad(self, features, padding = True, max_length = None,
            pad_to_multiple_of = None, return_tensors = None, **kwargs):
        width = max(len(f["input_ids"]) for f in features)
        if pad_to_multiple_of:
            width = -(-width // pad_to_multiple_of) * pad_to_multiple_of
        ids = [f["input_ids"] + [PAD_ID] * (width - len(f["input_ids"])) for f in features]
        out = {"input_ids": ids, "attention_mask": [[1] * len(f["input_ids"]) + [0] * (width - len(f["input_ids"]))
                                                    for f in features]}
        if return_tensors == "pt":
            out = {k: torch.tensor(v) for k, v in out.items()}
        return out


def _trainer(collator):
    trainer = StubTrainer(collator, Dataset.from_dict({"input_ids": [list(LONG_ROW), list(SHORT_ROW)]}))
    trainer.processing_class = PaddingTokenizer()
    return trainer


def _width(trainer):
    rows = [trainer.train_dataset[i] for i in range(len(trainer.train_dataset))]
    batch = trainer.data_collator(rows)
    assert batch["labels"].shape == batch["input_ids"].shape
    return batch["input_ids"].shape[1]


def _trl_collator(pad_to_multiple_of):
    cls = sft_trainer.DataCollatorForLanguageModeling
    fields = set(inspect.signature(cls.__init__).parameters)
    kwargs = {"pad_token_id": PAD_ID, "pad_to_multiple_of": pad_to_multiple_of}
    if "padding_free" in fields:         kwargs["padding_free"] = False
    if "completion_only_loss" in fields: kwargs["completion_only_loss"] = False
    return cls(**kwargs)


def test_trl_collator_pad_to_multiple_of_survives_the_swap():
    out = train_on_responses_only(_trainer(_trl_collator(16)), INSTRUCTION_PART, RESPONSE_PART)
    assert isinstance(out.data_collator, DataCollatorForSeq2Seq)
    assert out.data_collator.pad_to_multiple_of == 16
    assert _width(out) == 16


def test_trl_collator_without_pad_to_multiple_of_still_pads_to_longest():
    out = train_on_responses_only(_trainer(_trl_collator(None)), INSTRUCTION_PART, RESPONSE_PART)
    assert out.data_collator.pad_to_multiple_of is None
    assert _width(out) == len(LONG_ROW)


def test_a_non_trl_collator_attribute_is_not_carried():
    class Custom:
        pad_to_multiple_of = 16
        def __call__(self, features): return {}

    out = train_on_responses_only(_trainer(Custom()), INSTRUCTION_PART, RESPONSE_PART)
    assert out.data_collator.pad_to_multiple_of is None
