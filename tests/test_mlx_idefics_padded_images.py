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

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

if "mlx_simulation" in (getattr(mx, "__file__", "") or ""):
    pytest.skip("mlx.core is the tests/mlx_simulation stub, not real MLX",
                allow_module_level = True)

idefics3 = pytest.importorskip("mlx_vlm.models.idefics3.idefics3")

IMAGE_TOKEN, TOKENS_PER_IMAGE, DIM, SIZE = 9, 2, 4, 4


def _fake_model():
    model = SimpleNamespace(
        config = SimpleNamespace(image_token_index = IMAGE_TOKEN, vision_config = SimpleNamespace(patch_size = 2)),
        language_model = SimpleNamespace(embed_tokens = lambda ids: mx.zeros((*ids.shape, DIM))),
        # One feature row per image token, valued by the image's mean pixel, so a shifted merge shows.
        vision_model = lambda pixels, **_: (
            mx.broadcast_to(pixels.mean(axis = (1, 2, 3))[:, None, None], (pixels.shape[0], TOKENS_PER_IMAGE, DIM)),
        ),
        connector = lambda features: features,
    )
    model._prepare_inputs_for_multimodal = (
        lambda features, embeds, ids: idefics3.Model._prepare_inputs_for_multimodal(model, features, embeds, ids)
    )
    return model


def test_padded_image_slots_do_not_shift_later_rows_features():
    from unsloth_zoo.mlx.compile import install_mlx_compile_patches

    install_mlx_compile_patches()
    # Row 0 has one image plus a zero padding slot; row 1 has two images.
    pixel_values = mx.zeros((2, 2, 3, SIZE, SIZE))
    pixel_values[0, 0] = 1.0
    pixel_values[1, 0] = 2.0
    pixel_values[1, 1] = 3.0
    pixel_attention_mask = mx.ones((2, 2, SIZE, SIZE), dtype = mx.bool_)
    pixel_attention_mask[0, 1] = False
    row0 = [1] + [IMAGE_TOKEN] * TOKENS_PER_IMAGE + [1] * (2 * TOKENS_PER_IMAGE)
    row1 = [1] + [IMAGE_TOKEN] * (2 * TOKENS_PER_IMAGE) + [1] * TOKENS_PER_IMAGE
    input_ids = mx.array([row0, row1])

    embeds = idefics3.Model.get_input_embeddings(
        _fake_model(), input_ids, pixel_values, pixel_attention_mask = pixel_attention_mask,
    ).inputs_embeds

    merged = [[round(v, 4) for v in embeds[r, :, 0].tolist()] for r in range(2)]
    assert merged[0][1:1 + TOKENS_PER_IMAGE] == [1.0] * TOKENS_PER_IMAGE
    assert merged[1][1:1 + 2 * TOKENS_PER_IMAGE] == [2.0] * TOKENS_PER_IMAGE + [3.0] * TOKENS_PER_IMAGE
