# 设置最低的 CMake 版本要求
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 项目名称和支持的语言（C++ 和 CUDA）
project(CUDA_BLAS_Project LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 的标准版本
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# 包含头文件的目录（即 include 文件夹路径）
include_directories(include)

# 设置可执行文件输出路径为项目根目录下的 bin/ 目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# 自动搜索 目录下的所有 .cu 文件
file(GLOB SRC_BLASMM src/blasmm/*)
file(GLOB SRC_BLASVV src/blasvv/*)
file(GLOB SRC_BLASVM src/blasvm/*)

# 创建静态库（blasmm），包含所有 blasmm 目录下的 .cu 文件，设置头文件路径
add_library(blasmm STATIC ${SRC_BLASMM})
target_include_directories(blasmm PUBLIC include/blasmm)

add_library(blasvv STATIC ${SRC_BLASVV})
target_include_directories(blasvv PUBLIC include/blasvv)

add_library(blasvm STATIC ${SRC_BLASVM})
target_include_directories(blasvm PUBLIC include/blasvm)

# 创建一个可执行文件 main_exe
add_executable(main_exec main.cpp)

# 将库链接到 main_exec 可执行文件中
target_link_libraries(main_exec blasmm blasvv blasvm)

# 为调试目的启用详细的 Makefile 输出（可选）
# set(CMAKE_VERBOSE_MAKEFILE ON)
