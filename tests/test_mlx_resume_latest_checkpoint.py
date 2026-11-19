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

"""train(resume_from_checkpoint=True) resumes from the newest checkpoint, as in HF."""

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")


class _Stop(Exception):
    pass


def _trainer(output_dir):
    from unsloth_zoo.mlx.trainer import MLXTrainer

    class _Trainer(MLXTrainer):
        def __init__(self):
            self.args = SimpleNamespace(output_dir=str(output_dir))

        def _stop_request_generation(self):
            return 0

        def _close_active_batch_iterator(self):
            pass

        def _ensure_distributed(self):
            raise _Stop

    return _Trainer()


def _resume_target(output_dir, value):
    trainer = _trainer(output_dir)
    with pytest.raises(_Stop):
        trainer.train(resume_from_checkpoint=value)
    return trainer._resume_from_checkpoint


def test_true_picks_the_highest_numbered_checkpoint(tmp_path):
    for name in ("checkpoint-2", "checkpoint-10", "checkpoint-9", "checkpoint-best"):
        (tmp_path / name).mkdir()
    (tmp_path / "checkpoint-99").write_text("not a dir")
    assert _resume_target(tmp_path, True) == str(tmp_path / "checkpoint-10")


def test_false_starts_fresh_and_paths_pass_through(tmp_path):
    assert _resume_target(tmp_path, False) is None
    assert _resume_target(tmp_path, None) is None
    assert _resume_target(tmp_path, "some/dir") == "some/dir"


def test_true_without_checkpoints_raises(tmp_path):
    with pytest.raises(ValueError, match="No valid checkpoint found"):
        _trainer(tmp_path).train(resume_from_checkpoint=True)
