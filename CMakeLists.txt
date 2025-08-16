cmake_minimum_required(VERSION 3.18)
project(exp_GOL)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GCC 12를 CUDA 호스트 컴파일러로 사용 (GCC 13.3.0 호환성 문제 해결)
if(EXISTS "/usr/bin/gcc-12")
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-12)
    set(CMAKE_C_COMPILER /usr/bin/gcc-12)
    set(CMAKE_CXX_COMPILER /usr/bin/g++-12)
    message(STATUS "Using GCC 12 for CUDA compatibility")
endif()

# Find packages
find_package(Threads REQUIRED)

# CUDA 지원 시도
find_package(CUDAToolkit QUIET)
set(USE_CUDA FALSE)

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    
    # CUDA 언어 활성화 시도
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
        
        # CUDA 컴파일러 플래그
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-strict-aliasing")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=1650")
        
        set(USE_CUDA TRUE)
        message(STATUS "CUDA enabled successfully")
    else()
        message(WARNING "CUDA toolkit found but compiler not available")
    endif()
else()
    message(STATUS "CUDA toolkit not found - building header-only")
endif()

# Find cuDNN (only if CUDA is enabled)
set(USE_CUDNN FALSE)
if(USE_CUDA)
    # Try to find cuDNN manually
    find_path(CUDNN_INCLUDE_DIR
        NAMES cudnn.h
        PATHS /usr/include /usr/local/cuda/include
        NO_DEFAULT_PATH
    )
    
    find_library(CUDNN_LIBRARY
        NAMES cudnn libcudnn
        PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 /usr/local/cuda/lib
        NO_DEFAULT_PATH
    )
    
    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        set(USE_CUDNN TRUE)
        message(STATUS "cuDNN found:")
        message(STATUS "  Include: ${CUDNN_INCLUDE_DIR}")
        message(STATUS "  Library: ${CUDNN_LIBRARY}")
        
        # Get cuDNN version
        file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_VERSION_FILE)
        string(REGEX MATCH "define CUDNN_MAJOR ([0-9]+)" _ "${CUDNN_VERSION_FILE}")
        set(CUDNN_MAJOR_VERSION ${CMAKE_MATCH_1})
        string(REGEX MATCH "define CUDNN_MINOR ([0-9]+)" _ "${CUDNN_VERSION_FILE}")
        set(CUDNN_MINOR_VERSION ${CMAKE_MATCH_1})
        string(REGEX MATCH "define CUDNN_PATCHLEVEL ([0-9]+)" _ "${CUDNN_VERSION_FILE}")
        set(CUDNN_PATCH_VERSION ${CMAKE_MATCH_1})
        
        message(STATUS "  Version: ${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION}.${CUDNN_PATCH_VERSION}")
    else()
        message(WARNING "cuDNN not found")
    endif()
endif()

# Define source and header files (존재하는 파일만 포함)
if(USE_CUDA)
    set(SOURCE_FILES
        src/d_matrix_2.cu
        src/GOLdatabase_2.cu
    )
    message(STATUS "Including CUDA source files")
else()
    set(SOURCE_FILES)
    message(STATUS "Header-only mode (no CUDA sources)")
endif()

set(HEADER_FILES
    src/d_matrix_2.hpp
    src/GOLdatabase_2.hpp
    src/perceptron_2.hpp
    src/utility.hpp
)

# Create library
if(USE_CUDA)
    add_library(exp_GOL ${SOURCE_FILES} ${HEADER_FILES})
    set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(exp_GOL CUDA::cudart CUDA::curand CUDA::cublas)
    
    # Link cuDNN if available
    if(USE_CUDNN)
        target_link_libraries(exp_GOL ${CUDNN_LIBRARY})
        target_include_directories(exp_GOL PUBLIC ${CUDNN_INCLUDE_DIR})
        target_compile_definitions(exp_GOL PRIVATE CUDNN_ENABLED)
        message(STATUS "Linked cuDNN to library")
    endif()
    
    target_compile_features(exp_GOL PUBLIC cuda_std_17)
    target_compile_definitions(exp_GOL PRIVATE CUDA_ENABLED)
    target_include_directories(exp_GOL PUBLIC src)
    message(STATUS "Built CUDA-enabled library")
else()
    # Header-only interface library
    add_library(exp_GOL INTERFACE)
    target_include_directories(exp_GOL INTERFACE src)
    target_compile_definitions(exp_GOL INTERFACE CPU_ONLY)
    message(STATUS "Built header-only library")
endif()

# Link threading
if(USE_CUDA)
    target_link_libraries(exp_GOL Threads::Threads)
else()
    target_link_libraries(exp_GOL INTERFACE Threads::Threads)
endif()

# Set compile features
if(USE_CUDA)
    target_compile_features(exp_GOL PUBLIC cxx_std_20)
    target_compile_options(exp_GOL PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
else()
    target_compile_features(exp_GOL INTERFACE cxx_std_20)
endif()

# 실행파일 생성 코드단위.
#if(USE_CUDA)
#    if(EXISTS "${CMAKE_SOURCE_DIR}/model/genGOL.cu")
#        add_executable(genGOL model/genGOL.cu)
#        set_source_files_properties(model/genGOL.cu PROPERTIES LANGUAGE CUDA)
#        target_link_libraries(genGOL exp_GOL)
#        target_compile_features(genGOL PRIVATE cuda_std_17)
#        target_include_directories(genGOL PRIVATE src)
#        message(STATUS "Added genGOL executable")
#    else()
#        message(STATUS "genGOL executable skipped (source not found)")
#    endif()
#else()
#    message(STATUS "genGOL executable skipped (requires CUDA)")
#endif()

if(USE_CUDA)
    if(EXISTS "${CMAKE_SOURCE_DIR}/model/mercury.cu")
        add_executable(mercury model/mercury.cu)
        set_source_files_properties(model/mercury.cu PROPERTIES LANGUAGE CUDA)
        target_link_libraries(mercury exp_GOL)
        target_compile_features(mercury PRIVATE cuda_std_17)
        target_include_directories(mercury PRIVATE src)
        message(STATUS "Added mercury executable")
    else()
        message(STATUS "mercury executable skipped (source not found)")
    endif()
else()
    message(STATUS "mercury executable skipped (requires CUDA)")
endif()


