# Copyright 2026 FlagOS Contributors
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

import pytest
import torch

import flag_gems
from flag_gems.utils import dim_compress

from . import base, consts


_COPY_CHECK_DTYPES = (torch.float32,)


def _tensor_copy_info(inp, dim):
    contiguous_inp = inp.contiguous()
    contiguous_copies = inp.data_ptr() != contiguous_inp.data_ptr()

    info = {
        "shape": tuple(inp.shape),
        "stride": tuple(inp.stride()),
        "is_contiguous": inp.is_contiguous(),
        "contiguous_ptr_changed": contiguous_copies,
        "dim": dim,
    }

    if dim is not None:
        compressed_inp = dim_compress(inp, dim)
        info.update(
            {
                "dim_compress_shape": tuple(compressed_inp.shape),
                "dim_compress_stride": tuple(compressed_inp.stride()),
                "dim_compress_ptr_changed": inp.data_ptr()
                != compressed_inp.data_ptr(),
            }
        )

    return info


def _print_copy_info(op_name, dtype, args, kwargs, torch_result, gems_result, info):
    print("\n========== max copy check ==========")
    print(f"op_name:                    {op_name}")
    print(f"dtype:                      {dtype}")
    print(f"shape:                      {info['shape']}")
    print(f"stride:                     {info['stride']}")
    print(f"dim:                        {info['dim']}")
    print(f"is_contiguous:              {info['is_contiguous']}")
    print(f"inp.contiguous copies:      {info['contiguous_ptr_changed']}")
    if info["dim"] is not None:
        print(f"dim_compress shape:         {info['dim_compress_shape']}")
        print(f"dim_compress stride:        {info['dim_compress_stride']}")
        print(f"dim_compress copies:        {info['dim_compress_ptr_changed']}")
    print(f"args:                       {args}")
    print(f"kwargs:                     {kwargs}")
    print(f"torch result:               {torch_result}")
    print(f"flag_gems result:           {gems_result}")
    print("====================================")


@pytest.mark.max_copy_check
@pytest.mark.parametrize("op_name", ["max", "max_dim"])
def test_max_copy_check_for_shape_file(op_name):
    bench = base.UnaryReductionBenchmark(
        op_name=op_name, torch_op=torch.max, dtypes=consts.FLOAT_DTYPES
    )
    bench.init_user_config()

    dtypes = [dtype for dtype in bench.to_bench_dtypes if dtype in _COPY_CHECK_DTYPES]
    if not dtypes:
        dtypes = [bench.to_bench_dtypes[0]]

    for dtype in dtypes:
        for input_tuple in bench.get_input_iter(dtype):
            args, kwargs = bench.unpack_to_args_kwargs(input_tuple)
            inp = args[0]
            dim = args[1] if len(args) > 1 and isinstance(args[1], int) else None

            info = _tensor_copy_info(inp, dim)
            torch_result = bench.torch_op(*args, **kwargs)
            with flag_gems.use_gems(exclude=["zero_"]):
                gems_result = bench.torch_op(*args, **kwargs)

            _print_copy_info(
                op_name=op_name,
                dtype=dtype,
                args=[tuple(arg.shape) if torch.is_tensor(arg) else arg for arg in args],
                kwargs=kwargs,
                torch_result=torch_result,
                gems_result=gems_result,
                info=info,
            )

            if isinstance(torch_result, tuple):
                torch.testing.assert_close(gems_result.values, torch_result.values)
                torch.testing.assert_close(gems_result.indices, torch_result.indices)
            else:
                torch.testing.assert_close(gems_result, torch_result)
