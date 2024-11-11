#!/bin/bash

set -e

BASE_DIR="$(cd "$(dirname "$0")"; pwd)";
cd $BASE_DIR
ort_version="1.16.3"
ort_folder_name="onnxruntime-${ort_version}"
cd $ort_folder_name

rm -rf build

export CPATH="$BASE_DIR/../nlohmann/single_include"

# export CROSS_NAME=gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu
export CROSS_NAME=gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
export CROSS_ROOT=/opt/${CROSS_NAME}
export COMPILER_PREFIX=aarch64-none-linux-gnu-
export CPPFLAGS="-I${CROSS_ROOT}/include" 
export CFLAGS="-I${CROSS_ROOT}/include"
export AR=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar
export AS=${CROSS_ROOT}/bin/${COMPILER_PREFIX}as
export LD=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld
export RANLIB=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib
export CC=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc
export CXX=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++
export NM=${CROSS_ROOT}/bin/${COMPILER_PREFIX}nm


export CPUINFO_TARGET_PROCESSOR=arm64

./build.sh --config Release --parallel --arm64 --use_qnn --qnn_home=/opt/qcom/aistack/qnn/2.15.0.230926/ \
  --parallel --skip_tests --allow_running_as_root \
  --cmake_extra_defines CMAKE_POSITION_INDEPENDENT_CODE=ON \
  onnxruntime_BUILD_UNIT_TESTS=OFF \
  CMAKE_BUILD_TYPE=Release \
  CPUINFO_TARGET_PROCESSOR=arm64 \
  CMAKE_SYSTEM_PROCESSOR=aarch64 \
  CPUINFO_BUILD_BENCHMARKS=OFF \
  CPUINFO_RUNTIME_TYPE=static \
  CMAKE_SYSTEM_NAME=Linux \
  CMAKE_C_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc \
  CMAKE_CXX_COMPILER=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++ \
  CMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  CMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  CMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
  onnxruntime_CROSS_AARCH64COMPILING=ON \
  onnxruntime_CROSS_COMPILING=ON \
  onnxruntime_ENABLE_CPUINFO=ON

cd ..

rm -rf AARCH64 || true
mkdir AARCH64  || true

cp -rf ./$ort_folder_name/include ./AARCH64  || true
RELEASE="./$ort_folder_name/build/Linux/Release"  || true
cp $RELEASE/*.a ./AARCH64 2>>/dev/null  || true
cp $RELEASE/*.so ./AARCH64 2>>/dev/null  || true

shopt -s globstar  || true
cp -rf $RELEASE/_deps/*-build/**/*.a ./AARCH64 2>>/dev/null  || true
cp -rf $RELEASE/_deps/*-build/**/*.so ./AARCH64 2>>/dev/null  || true

printf "\nDone :) \n"
