#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <cublas_v2.h>   // CUBLAS 库
#include <cuda_runtime.h>
#include "blas.h"        // 统一头文件，包含自定义的 CUDA 算子

#define DATATYPE float

void print_help();
void print_performance(const std::string& operation, std::chrono::duration<double> duration);
bool verify_results(const float* h_ref, const float* h_res, int n);

//验证正确性并进行性能对比
void test_vector_add(int n);
void test_matrix_add(int width, int height);
void test_matrix_transpose(int width, int height);

// 主函数：允许选择要测试的算子
int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_help();
        return 1;
    }

    std::string operation = argv[1];

    if (operation == "vector_add") {
        int n = 100;
        if (argc !=4) std::cout << "Using default size " << n << std::endl;
        else if(std::string(argv[2]) == "--size") {
            n = std::atoi(argv[3]);
        }
        test_vector_add(n);
    }
    else if (operation == "matrix_add") {
        int width = 1024;
        int height = 1024;
        if (argc !=5) std::cout << "Using default size 1024 * 1024" << std::endl;
        else if (std::string(argv[2]) == "--matrix-size") {
            width = std::atoi(argv[3]);
            height = std::atoi(argv[4]);
        }
        test_matrix_add(width, height);
    }
    else if (operation == "matrix_transpose") {
        int width = 1024;
        int height = 1024;
        if (argc !=5) std::cout << "Using default size 1024 * 1024" << std::endl;
        else if (std::string(argv[2]) == "--matrix-size") {
            width = std::atoi(argv[3]);
            height = std::atoi(argv[4]);
        }
        test_matrix_transpose(width, height);
    }
    else {
        std::cout << "Unknown operation: " << operation << "\n";
        return 1;
    }

    return 0;
}

// 打印帮助信息
void print_help() {
    std::cout << "Usage: ./test.exe <operation> --size <n> or --matrix-size <width> <height>\n";
    std::cout << "Operations:\n";
    std::cout << "  vector_add: Test vector addition\n";
    std::cout << "  matrix_add: Test matrix addition\n";
    std::cout << "  matrix_transpose: Test matrix transpose\n";
}

// 打印性能
void print_performance(const std::string& operation, std::chrono::duration<double> duration) {
    std::cout << operation << " took: " << duration.count() * 1000 << " ms" << std::endl;
}

// 向量加法的 CUBLAS 性能测试
void test_vector_add(int n) {
    DATATYPE* h_a = (DATATYPE*)malloc(n * sizeof(DATATYPE));
    DATATYPE* h_b = (DATATYPE*)malloc(n * sizeof(DATATYPE));   // 自写算子 CUDA 结果
    DATATYPE* h_ref = (DATATYPE*)malloc(n * sizeof(DATATYPE)); // 参考结果
    DATATYPE alpha = static_cast<DATATYPE>(rand()) / RAND_MAX;
    // 初始化向量
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<DATATYPE>(rand()) / RAND_MAX;
        h_b[i] = static_cast<DATATYPE>(rand()) / RAND_MAX;
        h_ref[i] = h_b[i];  // CUBLAS的y值初始化
    }

    // 使用 cblas_saxpy 执行向量加法 y = a * x + y
    // 这里 a = 1.0， y = h_b, x = h_a
    auto start_cublas = std::chrono::high_resolution_clock::now();
    cublasHandle_t handle;
    cublasCreate(&handle);
    DATATYPE *d_a, *d_ref;
    cudaMalloc((void**)&d_a, n * sizeof(DATATYPE));
    cudaMalloc((void**)&d_ref, n * sizeof(DATATYPE));
    cublasSetVector(n, sizeof(DATATYPE), h_a, 1, d_a, 1); // H2D host to device
    cublasSetVector(n, sizeof(DATATYPE), h_ref, 1, d_ref, 1);
    #ifdef DATATYPE
    cublasStatus_t status = cublasSaxpy(handle, n, &alpha, d_a, 1, d_ref, 1);// h_ref = alpha * h_a + h_b
    #else
    cublasStatus_t status = cublasDaxpy(handle, n, &alpha, d_a, 1, d_ref, 1);// h_ref = alpha * h_a + h_b
    #endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS failed!" << status << std::endl;
        exit(EXIT_FAILURE);
    }
    cublasGetVector(n, sizeof(DATATYPE), d_ref, 1, h_ref, 1);
    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_ref);
    auto end_cublas = std::chrono::high_resolution_clock::now();
    print_performance("CUBLAS Vector Addition (CPU)", end_cublas - start_cublas);

    // 调用自定义 CUDA 向量加法函数
    auto start_cuda = std::chrono::high_resolution_clock::now();
    vector_add<DATATYPE>(h_a, alpha, h_b, n);  // 我的 CUDA 实现
    auto end_cuda = std::chrono::high_resolution_clock::now();
    print_performance("My CUDA Vector Addition", end_cuda - start_cuda);

    // 验证 CUDA 结果是否与 CBLAS 结果匹配
    if (verify_results(h_ref, h_b, n)) {
        std::cout << "CUDA vector addition is correct!" << std::endl;
    } else {
        std::cout << "CUDA vector addition has errors!" << std::endl;
    }

    // // 清理内存
    free(h_a);
    free(h_b);
    free(h_ref);
}

// 矩阵加法的 CUBLAS 性能测试
void test_matrix_add(int width, int height) {
    return;
    // int n = width * height;
    // float* h_a = (float*)malloc(n * sizeof(DATATYPE));
    // float* h_b = (float*)malloc(n * sizeof(DATATYPE));
    // float* h_c = (float*)malloc(n * sizeof(DATATYPE));

    // for (int i = 0; i < n; ++i) {
    //     h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    //     h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    // }

    // auto start = std::chrono::high_resolution_clock::now();
    // test_matrix_add(h_a, h_b, h_c, width, height);
    // auto end = std::chrono::high_resolution_clock::now();
    // print_performance("CUBLAS Matrix Addition", end - start);

    // free(h_a);
    // free(h_b);
    // free(h_c);
}

// 矩阵转置的 CUBLAS 性能测试
void test_matrix_transpose(int width, int height) {
    return;
    // int n = width * height;
    // float* h_a = (float*)malloc(n * sizeof(DATATYPE));
    // float* h_t = (float*)malloc(n * sizeof(DATATYPE));

    // for (int i = 0; i < n; ++i) {
    //     h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    // }

    // auto start = std::chrono::high_resolution_clock::now();
    // test_matrix_transpose(h_a, h_t, width, height);
    // auto end = std::chrono::high_resolution_clock::now();
    // print_performance("CUBLAS Matrix Transpose", end - start);

    // free(h_a);
    // free(h_t);
}

// 辅助函数，用于验证两个向量结果的正确性
bool verify_results(const float* h_ref, const float* h_res, int n) {
    float epsilon = 1e-5f; // 误差允许范围
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_ref[i] - h_res[i]) > epsilon) {
            return false;
        }
    }
    return true;
}