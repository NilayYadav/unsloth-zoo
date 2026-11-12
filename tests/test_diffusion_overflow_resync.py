# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from unsloth_zoo.diffusion_studio import visual_engine as V


def _server(lines):
    server = SimpleNamespace(p = SimpleNamespace(stdout = io.StringIO("".join(l + "\n" for l in lines))))
    server._send = lambda *a, **k: None
    return server


@pytest.mark.parametrize("err", ["ERR toolong 9000 4096", "ERR gen"])
def test_the_turn_after_a_refused_one_gets_its_own_reply(err):
    server = _server([err, "STATS prompt_n=1", "DONE", 'C 0 "second"', "DONE"])
    with pytest.raises(RuntimeError):
        V._generate_visual_once(server, [], 0, 1, None, None, None)
    assert V._generate_visual_once(server, [], 0, 1, None, None, None) == "second"


def test_an_error_without_a_done_does_not_swallow_the_next_reply():
    server = _server(["ERR parse bad json", 'C 0 "second"', "DONE"])
    with pytest.raises(RuntimeError):
        V._generate_visual_once(server, [], 0, 1, None, None, None)
    assert V._generate_visual_once(server, [], 0, 1, None, None, None) == "second"
