cmake_minimum_required(VERSION 3.10.2)

project(RuntimeLibrary)

# where is the nx-c source code located?
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DEPS_DIR ${CMAKE_CURRENT_LIST_DIR}/deps)

######################### customize when cross-compiling ###############################################################

# set COMPILER_PREFIX, for example, "" for default compiler, arm-linux- , or aarch64-linux- etc for cross compilers
# set generic build target
set(GENERIC_BUILD_TARGET AARCH64)
# set(CROSS_ROOT "/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu")
# set(CROSS_ROOT "/opt/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu")
set(CROSS_ROOT "/opt/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu")
set(COMPILER_PREFIX "aarch64-none-linux-gnu-")

set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_LINKER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

#--build: the machine you are building on
#--host: the machine you are building for
#--target: the machine that GCC will produce binary for
set(BUILD X86_64-linux-gnu)
set(HOST ${GENERIC_BUILD_TARGET}-linux)
set(TARGET ${GENERIC_BUILD_TARGET})

######################### webassembly wamr settings ####################################################################

# set make system
set(CMAKE_SYSTEM_NAME Linux)

######################## webassembly wamr library build ################################################################

# Reset default linker flags
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS " ")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS " ")

# compiling and linking flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wformat -Wformat-security -fdata-sections -ffunction-sections -Wl,--gc-sections")

####################### toolchain arguments ############################################################################

INCLUDE(CMakeForceCompiler)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -std=c99 -fno-common -Wall -Wextra")
set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -Wno-missing-field-initializers -fno-exceptions")
set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -Wno-attribute-warning -fdata-sections -ffunction-sections -Wl,--gc-sections ")

set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")
set(CMAKE_CPP_FLAGS " ${CMAKE_CPP_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")

message(STATUS "SDKTARGETSYSROOT=${SDKTARGETSYSROOT}")
message(STATUS "CROSS_ROOT=${CROSS_ROOT}")

set(CMAKE_AR ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar)
set(CMAKE_AS ${CROSS_ROOT}/bin/${COMPILER_PREFIX}as)
set(CMAKE_LD ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
set(CMAKE_RANLIB ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib)
set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_ASM_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_NM ${CROSS_ROOT}/bin/${COMPILER_PREFIX}NM)

# this is the file system root of the target
set(CMAKE_FIND_ROOT_PATH ${SDKTARGETSYSROOT})

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

######################## generic build settings ########################################################################

add_definitions(-DENABLE_PROFILING)

# set some basic c, c++ and linker flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(LINKER_FLAGS "${LINKER_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")

# add all source files
file(GLOB_RECURSE SRC
        ${SRC_DIR}/runtime_utils.c
        ${SRC_DIR}/runtime_core.c
        ${DEPS_DIR}/utils/queue.c
        ${DEPS_DIR}/utils/logger.c
)
######################## main bin build settings #######################################################################

# binary target
add_library(RuntimeLibrary SHARED ${SRC})
target_compile_options(RuntimeLibrary PUBLIC -Werror)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        target_compile_options(RuntimeLibrary PUBLIC
                -Ofast -s
        )
endif()

# debugging on/off RuntimeLibrary
if(NX_DEBUG)
        target_compile_definitions(RuntimeLibrary PRIVATE NX_DEBUG_PRINT)
endif()

# target include / link dirs
target_include_directories(RuntimeLibrary PUBLIC ${INCLUDE_DIR})
target_include_directories(RuntimeLibrary PUBLIC ${DEPS_DIR}/utils)

set(ONNX_RUNTIME_DIR "${DEPS_DIR}/onnxruntime")
set(CPUINFO_DIR "${DEPS_DIR}/cpuinfo")
set(QNN_DIR "${DEPS_DIR}/qnn")

target_include_directories(RuntimeLibrary PUBLIC ${CPUINFO_DIR}/include)
target_link_directories(RuntimeLibrary PUBLIC "${CPUINFO_DIR}/${GENERIC_BUILD_TARGET}")
target_link_libraries(RuntimeLibrary PUBLIC cpuinfo)

target_include_directories(RuntimeLibrary PUBLIC ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}/include)
target_include_directories(RuntimeLibrary PUBLIC ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}/include/onnxruntime/core/session)
target_link_directories(RuntimeLibrary PUBLIC "${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}")
target_link_libraries(
        RuntimeLibrary
        PUBLIC

        onnxruntime_session
        onnxruntime_optimizer
        onnxruntime_providers
        onnxruntime_providers_qnn
        onnxruntime_util
        onnxruntime_framework
        onnxruntime_graph
        onnxruntime_mlas
        onnxruntime_common
        onnxruntime_flatbuffers

        absl_bad_optional_access
        absl_bad_variant_access
        absl_base
        absl_city
        absl_civil_time
        absl_cord
        absl_cord_internal
        absl_cordz_functions
        absl_cordz_handle
        absl_cordz_info
        # absl_crc32c
        # absl_crc_cord_state
        # absl_crc_cpu_detect
        # absl_crc_internal
        absl_debugging_internal
        absl_demangle_internal
        absl_exponential_biased
        # absl_flags_commandlineflag
        # absl_flags_commandlineflag_internal
        # absl_flags_config
        # absl_flags_internal
        # absl_flags_marshalling
        # absl_flags_private_handle_accessor
        # absl_flags_program_name
        # absl_flags_reflection
        # absl_graphcycles_internal
        absl_hash
        absl_hashtablez_sampler
        absl_int128
        # absl_kernel_timeout_internal
        absl_log_severity
        absl_low_level_hash
        absl_malloc_internal
        absl_raw_hash_set
        absl_raw_logging_internal
        absl_spinlock_wait
        absl_stacktrace
        # absl_str_format_internal
        # absl_string_view
        absl_strings
        absl_strings_internal
        absl_symbolize
        absl_synchronization
        absl_throw_delegate
        absl_time
        absl_time_zone
        cpuinfo
        flatbuffers
        nsync_cpp
        onnx
        onnx_proto
        protobuf-lite
        clog

        pthread
        stdc++
)

# Copy all files inside QNN_DIR to the build directory
add_custom_command(TARGET RuntimeLibrary POST_BUILD
        COMMAND cp ${QNN_DIR}/${GENERIC_BUILD_TARGET}/* ${CMAKE_CURRENT_BINARY_DIR}/
)
add_custom_command(TARGET RuntimeLibrary POST_BUILD
        COMMAND cp ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}/*.so ${CMAKE_CURRENT_BINARY_DIR}/
)
