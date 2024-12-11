#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import numpy
import PIL
import torch

from lerobot.common.datasets.video_utils import encode_video_frames


def concatenate_episodes(ep_dicts):
    data_dict = {}

    keys = ep_dicts[0].keys()
    for key in keys:
        if torch.is_tensor(ep_dicts[0][key][0]):
            data_dict[key] = torch.cat([ep_dict[key] for ep_dict in ep_dicts])
        else:
            if key not in data_dict:
                data_dict[key] = []
            for ep_dict in ep_dicts:
                for x in ep_dict[key]:
                    data_dict[key].append(x)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def save_images_concurrently(imgs_array: numpy.array, out_dir: Path, max_workers: int = 4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_image(img_array, i, out_dir):
        img = PIL.Image.fromarray(img_array)
        img.save(str(out_dir / f"frame_{i:06d}.png"), quality=100)

    num_images = len(imgs_array)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        [executor.submit(save_image, imgs_array[i], i, out_dir) for i in range(num_images)]


def get_default_encoding() -> dict:
    """Returns the default ffmpeg encoding parameters used by `encode_video_frames`."""
    return {
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
        "preset": "medium",
        "crf": 23,
    }

def get_default_encoding_old() -> dict:
    """Returns the default ffmpeg encoding parameters used by `encode_video_frames`."""
    signature = inspect.signature(encode_video_frames)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and k in ["vcodec", "pix_fmt", "g", "crf"]
    }


def check_repo_id(repo_id: str) -> None:
    if len(repo_id.split("/")) != 2:
        raise ValueError(
            f"""`repo_id` is expected to contain a community or user id `/` the name of the dataset
            (e.g. 'lerobot/pusht'), but contains '{repo_id}'."""
        )


def quat2euler(quat):
    """
    Convert a quaternion or an array of quaternions to Euler angles.
    The Euler angles are returned in the order: roll (X), pitch (Y), yaw (Z).

    This function expects quaternions in the [w, x, y, z] format.

    Parameters:
    -----------
    quat : array_like
        Quaternion(s) to convert. Shape can be (4,) for a single quaternion
        or (..., 4) for multiple quaternions. The expected order is [w, x, y, z].

    Returns:
    --------
    euler : ndarray
        Euler angles in radians. Shape is (..., 3).
    """
    # Define a small epsilon to handle numerical precision
    _EPS4 = numpy.finfo(float).eps * 4.0

    # Convert input to numpy array with float64 dtype
    quat = numpy.asarray(quat, dtype=numpy.float64)

    # Ensure the quaternion has the correct shape
    if quat.ndim == 1:
        quat = quat[numpy.newaxis, :]  # Convert to 2D array for consistency
    elif quat.ndim > 2 or quat.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape {quat.shape}. Expected shape (..., 4).")

    # Create a Rotation object from quaternions
    rotation = R.from_quat(quat, scalar_first=True)

    # Convert to rotation matrices
    mat = rotation.as_matrix()  # Shape (..., 3, 3)

    # Compute cy = sqrt(mat[...,2,2]^2 + mat[...,1,2]^2)
    cy = numpy.sqrt(mat[..., 2, 2] ** 2 + mat[..., 1, 2] ** 2)

    # Determine where cy is significant to avoid division by zero
    condition = cy > _EPS4

    # Initialize Euler angles array
    if len(mat.shape) == 3:
        euler = numpy.empty((mat.shape[0], 3), dtype=numpy.float64)
    else:
        euler = numpy.empty(mat.shape[:-1] + (3,), dtype=numpy.float64)

    # Compute yaw (Z axis rotation)
    euler[..., 2] = numpy.where(
        condition,
        -numpy.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -numpy.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )

    # Compute pitch (Y axis rotation)
    euler[..., 1] = -numpy.arctan2(mat[..., 0, 2], cy)

    # Compute roll (X axis rotation)
    euler[..., 0] = numpy.where(
        condition,
        -numpy.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
        0.0  # Gimbal lock: roll is set to zero
    )

    # If the input was a single quaternion, return a 1D array
    if euler.shape[0] == 1:
        return euler[0]

    return euler
