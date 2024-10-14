#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath>
#include <cublas_v2.h>   // CUBLAS 库
#include <cuda_runtime.h>
#include "blas.h"        // 统一头文件，包含自定义的 CUDA 算子
#include <unistd.h>
void print_help();
void print_vm(float*, int, int);
bool verify_results(const float* h_ref, const float* h_res, int n);
void init_matrix(int size_A, float *h_a);
void cublasTest(cublasStatus_t status);

//验证正确性并进行性能对比
void test_vector_add(int n);
void test_gemm(int m, int n, int k, int iter);

// 主函数：允许选择要测试的算子
int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_help();
        return 1;
    }

    std::string operation = argv[1];

    if (operation == "vector_add") {
        int n = 10;
        if (argc !=4) printf("Using default size %d.\n", n);
        else if(std::string(argv[2]) == "--size") {
            n = std::atoi(argv[3]);
        }
        test_vector_add(n);
    }
    else if (operation == "gemm") {
        int m = 1024;
        int n = 1024;
        int k = 1024;
        int iter = 10;
        if (argc !=6) printf("Using default size %d * %d * %d.\n", m, n, k);
        else if (std::string(argv[2]) == "--matrix-size") {
            m = std::atoi(argv[3]);
            n = std::atoi(argv[4]);
            k = std::atoi(argv[5]);
        }
        test_gemm(m, n, k, iter);
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
    std::cout << "  gemm: Test sgemm\n";
}

// 向量加法的 CUBLAS 性能测试
void test_vector_add(int n) {
    int size = n * sizeof(float);
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);   // 自写算子 CUDA 结果
    float* h_ref = (float*)malloc(size); // 参考结果
    float alpha = static_cast<float>(rand()) / RAND_MAX;
    alpha = 1;

    // 初始化向量
    init_matrix(n, h_a);
    init_matrix(n, h_b);
    memcpy(h_ref, h_b, size);

    // print_vm(h_b, 1, n);
    // 使用 cublas_saxpy 执行向量加法 y = a * x + y
    // 这里 a = 1.0， y = h_b, x = h_a
    float *d_a, *d_ref;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_ref, n * sizeof(float));
    cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1); // H2D host to device
    cublasSetVector(n, sizeof(float), h_ref, 1, d_ref, 1);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);    
    cublasStatus_t status = cublasSaxpy(handle, n, &alpha, d_a, 1, d_ref, 1);// h_ref = alpha * h_a + h_b
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cublasTest(status);
    cublasGetVector(n, sizeof(float), d_ref, 1, h_ref, 1);
    float time_acc = 0.f;
    cudaEventElapsedTime(&time_acc, start, stop);
    printf("%f ms used.\n", time_acc);
    long ops = (long)n * 2;
    double gops = ((double)ops / 1e9) / ((double)time_acc / 1e3);
    printf("CUBLAS Vector ADD: %f Gops\n", gops);

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_ref);

    // 调用自定义 CUDA 向量加法函数
    vector_add<float>(h_a, alpha, h_b, n);  // 我的 CUDA 实现

    // 验证 CUDA 结果是否与 CBLAS 结果匹配
    printf("My CUDA vector ADD is %s!\n", 
            verify_results(h_ref, h_b, n) ? "true":"false");
    // printf("%f\n", alpha);
    // print_vm(h_a, 1, n);
    // print_vm(h_b, 1, n);
    // print_vm(h_ref, 1, n);
    // // 清理内存
    free(h_a);
    free(h_b);
    free(h_ref);
}

#define CHECK_CUDA(call) {                                          \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",\
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}
// gemm 性能测试
void test_gemm(int m, int n, int k, int iter) {
    int size_A = n * m * sizeof(float);
    int size_B = n * k * sizeof(float);
    int size_C = m * k * sizeof(float);
    float* h_a = (float*)malloc(size_A);
    float* h_b = (float*)malloc(size_B);
    float* h_c = (float*)malloc(size_C);   // 自写算子 CUDA 结果
    float* h_ref = (float*)malloc(size_C); // 参考结果
    float alpha = static_cast<float>(rand()) / RAND_MAX;
    float beta = static_cast<float>(rand()) / RAND_MAX;
    alpha = 1;
    beta = 0;
    // 初始化向量
    init_matrix(m * n, h_a);
    init_matrix(n * k, h_b);
    init_matrix(m * k, h_c);
    memcpy(h_ref, h_c, size_C);  
    // print_vm(h_a, m, n);
    // print_vm(h_b, n, k);
    
   
    // 使用 cublas
    cublasStatus_t status;
    float *d_a, *d_b, *d_ref;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size_A));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size_B));
    CHECK_CUDA(cudaMalloc((void**)&d_ref, size_C));
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice)); // H2D host to device
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ref, h_ref, size_C, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //  warmup
    status = cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           k, m, n, 
                                           &alpha,
                                           d_b, k, 
                                           d_a, n, 
                                           &beta, 
                                           d_ref, k);
    cublasTest(status);
    cublasGetVector(m * k, sizeof(float), d_ref, 1, h_ref, 1);

    float time_acc = 0.f;
    float time;
    long ops = (long)m * n * k * 2;
    double gops;
    printf("CUBLAS performance:\n ROUND      time          GFLOPS\n");
    for(int i = 0; i < iter; i++){
        cudaEventRecord(start, 0);
        status = cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            k, m, n, 
                                            &alpha,
                                            d_b, k, 
                                            d_a, n, 
                                            &beta, 
                                            d_ref, k);// h_ref = alpha * h_a + h_b
        cudaDeviceSynchronize(); 
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        // cublasTest(status);
        cudaEventElapsedTime(&time, start, stop);
        time_acc += time;
        gops = ((double)ops / 1e9) / ((double)time / 1e3);
        printf("Round %d: %f ms | %f\n", i, time, gops);
    }
    time = time_acc / iter; 
    gops = ((double)ops / 1e9) / ((double)time / 1e3);
    printf("Average: %f ms | %f\n", time, gops);

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ref);

    // 调用自定义 CUDA gemm函数
    gemm(h_a, h_b, h_c, alpha, beta, m, n, k, iter);  // 我的 CUDA 实现

    // print_vm(h_ref, 12, 12);
    // printf("------------------------------------------\n");
    // print_vm(h_c, 12, 12);
    // 验证 CUDA 结果是否与 CBLAS 结果匹配
    printf("My CUDA gemm is %s!\n", 
            verify_results(h_ref, h_c, m * k) ? "true":"false");

    // // 清理内存
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);
}

void cublasTest(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS failed with error code: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
    else
        std::cout << "Final success!" << std::endl;
}

void init_matrix(int size_A, float *h_a)
{
    for (int i = 0; i < size_A; ++i)
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
}

// 辅助函数，用于验证两个向量结果的正确性
bool verify_results(const float* h_ref, const float* h_res, int n) {
    float epsilon = 2e-3f; // 误差允许范围 
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_ref[i] - h_res[i]) > epsilon) {
            printf("wrong! %f\n",std::fabs(h_ref[i] - h_res[i]) );
            return false;
        }
    }
    return true;
}

void print_vm(float *A, int row, int col) {
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++)
            printf("%f ", A[i * row + j]);
        printf("\n");
    }
}