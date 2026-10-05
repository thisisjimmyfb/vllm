# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for miscellaneous utilities."""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.platforms import current_platform


def test_convert_fp8_opcheck():
    data = torch.randn((256, 256), dtype=torch.float32, device="cuda")
    result = torch.empty_like(data, dtype=torch.float8_e4m3fn)
    opcheck(torch.ops._C_cache_ops.convert_fp8, (result, data, 1.0, "fp8"))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only supported for CUDA")
def test_cuda_utils_opcheck():
    cuda_dev_attr_multiprocessor_count = 16
    opcheck(
        torch.ops._C_cuda_utils.get_device_attribute,
        (cuda_dev_attr_multiprocessor_count, 0),
    )
    opcheck(
        torch.ops._C_cuda_utils.get_max_shared_memory_per_block_device_attribute,
        (0,),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only supported for CUDA")
def test_get_device_attribute_caches_per_attribute():
    """Each attribute returns its own value, not the first one queried."""
    cuda_dev_attr_multiprocessor_count = 16
    props = torch.cuda.get_device_properties(0)
    assert (
        torch.ops._C_cuda_utils.get_device_attribute(
            cuda_dev_attr_multiprocessor_count, 0
        )
        == props.multi_processor_count
    )
    assert (
        torch.ops._C_cuda_utils.get_max_shared_memory_per_block_device_attribute(0)
        == props.shared_memory_per_block_optin
    )
