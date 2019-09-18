"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from te import tvm
import te.lang.cce
import topi

def sign_cce(data_shape, dtype, kernel_name = "cce_sign", need_build = False, need_print = False):
    """
                                  x * 32768
    algrithm: sign = round(-------------------------)
                            2 ** (-15) + |x * 32768|   

    calculating data type is float16
    
    Parameters
    ----------
    data_shape : shape of data

    dtype : the data type, assume src_dtype equals dst_dtype, only support float16, float32, int32

    kernel_name : cce kernel name, default value is "cce_sign"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None
        
    """
    fp16_max = tvm.const(32768, dtype="float16")
    fp16_min = tvm.const(2 ** (-15), dtype = "float16")
    data = tvm.placeholder(data_shape, name="data", dtype=dtype)
    with tvm.target.cce():
        data_tmp = te.lang.cce.cast_to(data, "float16")
        new_data = te.lang.cce.vmuls(data_tmp, fp16_max)
        tmp2 = te.lang.cce.vabs(new_data)
        anuminate = te.lang.cce.vadds(tmp2, fp16_min)
        fp16_res = te.lang.cce.vmul(new_data, te.lang.cce.vrec(anuminate))
        res_tmp = te.lang.cce.round(fp16_res)
        res = te.lang.cce.cast_to(res_tmp, dtype)
        sch = topi.generic.auto_schedule(res)

    config = {"print_ir" : need_print,
             "need_build" : need_build,
             "name" : kernel_name,
             "tensor_list" : [data, res]}

    te.lang.cce.cce_build_code(sch, config)

if __name__ == "__main__":
    sign_cce((2, 4), "float16")
