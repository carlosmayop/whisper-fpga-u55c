cmake_minimum_required(VERSION 3.16)
project(whisper_fpga_encoder CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#  Mode selection 
# -DFPGA_STUB=ON  → CPU stub (no XRT, no hardware needed)
# -DFPGA_STUB=OFF → Real XRT implementation (default)
option(FPGA_STUB "Build CPU stub instead of XRT implementation" OFF)

set(HLS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

if(FPGA_STUB)
    #  Vitis HLS include path (for ap_int.h, hls_half.h, etc.) 
    if(NOT DEFINED VITIS_ROOT)
        if(DEFINED ENV{XILINX_VITIS})
            set(VITIS_ROOT "$ENV{XILINX_VITIS}")
        else()
            set(VITIS_ROOT "/opt/Xilinx/Vitis/2025.2")
        endif()
    endif()
    set(HLS_INCLUDE_DIR "${VITIS_ROOT}/include")

    if(NOT EXISTS "${HLS_INCLUDE_DIR}/ap_int.h")
        message(FATAL_ERROR "HLS headers not found at ${HLS_INCLUDE_DIR}. "
                            "Set -DVITIS_ROOT=/path/to/Vitis")
    endif()
    message(STATUS "FPGA_STUB mode: using HLS headers from ${HLS_INCLUDE_DIR}")

    #  Shared library (stub) 
    add_library(whisper_fpga_encoder SHARED
        whisper_fpga_encoder_stub.cpp
        # HLS kernel sources compiled as plain C++
        ${HLS_SRC_DIR}/encoder_top.cpp
        ${HLS_SRC_DIR}/conv_layers.cpp
        ${HLS_SRC_DIR}/layer_norm.cpp
        ${HLS_SRC_DIR}/matmul.cpp
        ${HLS_SRC_DIR}/multi_head_att.cpp
    )

    target_include_directories(whisper_fpga_encoder PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HLS_SRC_DIR}
        ${HLS_INCLUDE_DIR}
    )

    # HLS simulation runtime libraries (provide xip_fpo_* and hls::exp/tanh/etc.)
    set(HLS_CSIM_LIB_DIR  "${VITIS_ROOT}/../lnx64/lib/csim")
    set(HLS_MODELCOMP_DIR "${VITIS_ROOT}/../Model_Composer/lib/lnx64.o")

    target_link_directories(whisper_fpga_encoder PRIVATE
        ${HLS_CSIM_LIB_DIR}
        ${HLS_MODELCOMP_DIR}
    )
    target_link_libraries(whisper_fpga_encoder PRIVATE
        hlsmc++-CLANG39
        fpo_v70
    )

    # Embed RPATH so the .so finds these libs at runtime without LD_LIBRARY_PATH
    set_target_properties(whisper_fpga_encoder PROPERTIES
        BUILD_RPATH "${HLS_CSIM_LIB_DIR}:${HLS_MODELCOMP_DIR}"
        INSTALL_RPATH "${HLS_CSIM_LIB_DIR}:${HLS_MODELCOMP_DIR}"
    )

    target_compile_options(whisper_fpga_encoder PRIVATE -Wall -O2)

else()
    #  XRT (Xilinx Runtime) 
    if(NOT DEFINED XRT_ROOT)
        set(XRT_ROOT "/opt/xilinx/xrt")
    endif()

    find_path(XRT_INCLUDE_DIR
        NAMES xrt/xrt_device.h
        HINTS "${XRT_ROOT}/include"
        REQUIRED
    )
    find_library(XRT_COREUTIL_LIB
        NAMES xrt_coreutil
        HINTS "${XRT_ROOT}/lib"
        REQUIRED
    )

    message(STATUS "XRT include: ${XRT_INCLUDE_DIR}")
    message(STATUS "XRT lib:     ${XRT_COREUTIL_LIB}")

    #  Shared library (XRT) 
    add_library(whisper_fpga_encoder SHARED
        whisper_fpga_encoder.cpp
    )

    target_include_directories(whisper_fpga_encoder PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${XRT_INCLUDE_DIR}
    )

    target_link_libraries(whisper_fpga_encoder PUBLIC
        ${XRT_COREUTIL_LIB}
        uuid
        dl
    )

    target_compile_options(whisper_fpga_encoder PRIVATE -Wall -O2)

endif()

#  Optional: standalone test binary 
option(BUILD_FPGA_TEST "Build standalone FPGA encoder test" OFF)
if(BUILD_FPGA_TEST)
    add_executable(fpga_test fpga_test.cpp)
    target_include_directories(fpga_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(fpga_test PRIVATE whisper_fpga_encoder)
endif()
