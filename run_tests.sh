#!/bin/bash

clang++ tests/cpp/ir/ir_builder_test.cpp \
    -std=c++17 \
    -Iexternal/googletest/googletest/include \
    -I. \
    -Iexternal/spdlog/include \
    -Iexternal/eigen \
    -DTI_INCLUDED \
    -Lbuild \
    -ltaichi_core_static \
    -Lpython/taichi/_lib/runtime \
    -lruntime_arm64
