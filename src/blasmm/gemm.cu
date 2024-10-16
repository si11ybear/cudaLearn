#include <cuda_runtime.h>
#include <stdio.h>
#include <typeinfo>

// CUDA error checking macro
#define CHECK_CUDA(call) {                                          \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",\
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

// Templated CUDA kernel for GEMM: C = alpha * A * B + beta * C
__global__ void gemm_kernel(float *A, float *B, float *C, float alpha, float beta, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float value = 0;
        for (int j = 0; j < n; j++) {
            value += A[row * n + j] * B[j * k + col];
        }
        C[row * k + col] = alpha * value + beta * C[row * k + col];
    }
}

// Templated GEMM function
void gemm(const float *h_A, const float *h_B, float *h_C, float alpha, float beta, int m, int n, int k, int iter) {
    // Calculate sizes
    int size_A = m * n * sizeof(float);
    int size_B = n * k * sizeof(float);
    int size_C = m * k * sizeof(float);

    // Device memory pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory on the device (GPU)
    CHECK_CUDA(cudaMalloc((void **)&d_A, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size_B));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size_C));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    // Define thread block and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //  warmup
    gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, alpha, beta, m, n, k);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost)); //注意回送数据的时机，现在要求beta = 0

    float time_acc = 0.f;
    float time;
    long ops = (long)m * n * k * 2;
    double gops;
    printf("My GEMM performance:\n ROUND      time          GFLOPS\n");
    for(int i = 0; i < iter; i++){
        cudaEventRecord(start, 0);
        gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, alpha, beta, m, n, k);
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

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}