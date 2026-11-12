# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# SPDX-License-Identifier: AGPL-3.0-or-later

# The visual server checks that an explicit MAXTOK's canvas buffer fits and
# otherwise auto-sizes no larger than it, so the requested value is passed on.

from __future__ import annotations

import pytest

from unsloth_zoo.diffusion_studio.visual_engine import _canvas_maxtok


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        (-1, 0),
        (1, 1),
        (4096, 4096),
        (8192, 8192),
        (8193, 8193),
        (24576, 24576),
        (65536, 65536),
    ],
)
def test_canvas_maxtok_passes_the_requested_value(value, expected):
    assert _canvas_maxtok(value) == expected


def test_canvas_maxtok_coerces_str():
    assert _canvas_maxtok("4096") == 4096
    assert _canvas_maxtok("24576") == 24576
