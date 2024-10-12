cmake_minimum_required(VERSION 3.14)
project(1_4_vectorAddWithBlas)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
link_directories("/usr/local/cuda-10.0/lib64") # libcublas.so 所在文件夹路径

cuda_add_executable(1_4_vectorAddWithBlas main.cpp)
target_link_libraries(1_4_vectorAddWithBlas cublas) # 链接库

