 # ============================================================================
 #
 # Copyright (C) 2019, Huawei Technologies Co., Ltd. All Rights Reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions are met:
 #
 #   1 Redistributions of source code must retain the above copyright notice,
 #     this list of conditions and the following disclaimer.
 #
 #   2 Redistributions in binary form must reproduce the above copyright notice,
 #     this list of conditions and the following disclaimer in the documentation
 #     and/or other materials provided with the distribution.
 #
 #   3 Neither the names of the copyright holders nor the names of the
 #   contributors may be used to endorse or promote products derived from this
 #   software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 # POSSIBILITY OF SUCH DAMAGE.
 # ============================================================================
export DDK_PATH=/home/ascend/tools/che/ddk/ddk/
export SLOG_PRINT_TO_STDOUT=1 && export PATH=${PATH}:$DDK_PATH/uihost/toolchains/ccec-linux/bin/ && export LD_LIBRARY_PATH=$DDK_PATH/uihost/lib/ && export TVM_AICPU_LIBRARY_PATH=$DDK_PATH/uihost/lib/:$DDK_PATH/uihost/toolchains/ccec-linux/aicpu_lib && export TVM_AICPU_INCLUDE_PATH=$DDK_PATH/include/inc/tensor_engine && export PYTHONPATH=$DDK_PATH/site-packages && export TVM_AICPU_OS_SYSROOT=$DDK_PATH/uihost/toolchains/aarch64-linux-gcc6.3/sysroot
