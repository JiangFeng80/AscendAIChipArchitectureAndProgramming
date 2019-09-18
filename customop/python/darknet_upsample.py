# pylint: disable=invalid-name, too-many-locals, too-many-statements
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

darknet upsample
"""

from te import tvm
from te.platform.cce_build import build_config
from topi.cce import util

def darknet_upsample_cce(shape, dtype, size, data_format="channels_last",
                         kernel_name="cce_darknet_upsample", need_build=False, need_print=False):
    """
    Operation and Schedule for upsample

    Parameters
    ----------
    shape: tuple
     input of Tensor shape (4D or 5D)

    dtype: str
     support: ["float16", "float32", "int32", "int8", "uint8"]

    size: tuple
     the scale of each axis (4D or 5D)

    data_format: str
     one of `channels_last` (default) or `channels_first`.
     For example(4D): The ordering of the dimensions in the inputs.
     `channels_last` corresponds to inputs with shape
     `(batch, height, width, channels)` while `channels_first`
     corresponds to inputs with shape
     `(batch, channels, height, width)`.

    kernel_name : cce kernel name, default value is "cce_darknet_upsample"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None
    """
    inp_dtype = dtype.lower()
    check_list = ["float16", "float32", "int32", "int8", "uint8"]
    if not (inp_dtype in check_list):
        raise RuntimeError(
            "upsample only support %s while dtype is %s" % (",".join(check_list), dtype))

    util.check_shape_rule(shape)

    if len(size) != 2:
        raise RuntimeError("the len must be 2 dmin while len(size): %d" % len(size))

    shape_size = len(shape)
    if not (shape_size == 4 or shape_size == 5):
        raise RuntimeError("upsample only support 4D or 5D while len(shape):%d" % len(shape))

    input_tensor = tvm.placeholder(shape, name="input_tensor", dtype=inp_dtype)

    Res = None
    if shape_size == 5:
        # shape_size == 5 D-sepecial (N, C1, H, W, C0)
        output_shape = (shape[0], shape[1], shape[2]*size[0], shape[3]*size[1], shape[4])
        Res = tvm.compute(output_shape,
                          lambda n, c0, h, w, c: input_tensor[n, c0, h // size[0], w // size[1], c])
    else:
        if data_format == "channels_last":
            output_shape = (shape[0], shape[1]*size[0], shape[2]*size[1], shape[3])
            Res = tvm.compute(output_shape,
                              lambda n, h, w, c: input_tensor[n, h // size[0], w // size[1], c])
        elif data_format == "channels_first":
            output_shape = (shape[0], shape[1], shape[2]*size[0], shape[3]*size[1])
            Res = tvm.compute(output_shape,
                              lambda n, c, h, w: input_tensor[n, c, h // size[0], w // size[1]])
        else:
            raise RuntimeError("upsample only support channels_last|channels_first "
                               "while input type %s" % data_format)

    s = tvm.create_schedule(Res.op)
    if need_print:
        with build_config:
            print(tvm.lower(s, [input_tensor, Res], simple_mode=True))

    if need_build:
        with build_config:
            tvm.build(s, [input_tensor, Res], "cce", name=kernel_name)
