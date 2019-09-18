# Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details at
# http://www.apache.org/licenses/LICENSE-2.0
#
ifeq ($(DDK_PATH),)
TOPDIR := ../../..
else
TOPDIR := $(DDK_PATH)
endif

LOCAL_DIR  := ./
INCLUDE_DIR := $(TOPDIR)/include

LOCAL_MODULE_NAME := customop_app

CC_FLAGS := -std=c++11 -Wl,-rpath=$(TOPDIR)/uihost/lib

local_src_files := \
    $(LOCAL_DIR)/main.cpp

local_inc_dirs := \
    $(LOCAL_DIR) \
    $(INCLUDE_DIR)/inc \
    $(INCLUDE_DIR)/third_party/protobuf/include

local_shared_libs_dirs := \
    $(TOPDIR)/uihost/lib \

local_shared_libs := \
    hiai_common mmpa c_sec  protobuf slog hiai_server_cmodel

include ./cc_rule.mk
