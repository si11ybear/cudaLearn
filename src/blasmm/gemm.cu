#include <cuda_runtime.h>
#include <stdio.h>
#include <typeinfo>

// Templated CUDA kernel for GEMM: C = alpha * A * B + beta * C
template <typename T>
__global__ void gemm_kernel(T *A, T *B, T *C, T alpha, T beta, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        T value = 0;
        for (int k = 0; k < A_cols; k++) {
            value += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = alpha * value + beta * C[row * B_cols + col];
    }
}

// Templated GEMM function
template <typename T>
void gemm(const T *h_A, const T *h_B, T *h_C, T alpha, T beta, int A_rows, int A_cols, int B_cols) {
    // Calculate sizes
    int size_A = A_rows * A_cols * sizeof(T);
    int size_B = A_cols * B_cols * sizeof(T);
    int size_C = A_rows * B_cols * sizeof(T);

    // Device memory pointers
    T *d_A, *d_B, *d_C;
    printf("here\n");

    // Allocate memory on the device (GPU)
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Define thread block and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((B_cols + blockSize.x - 1) / blockSize.x, (A_rows + blockSize.y - 1) / blockSize.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start, 0);

    // Launch the kernel
    gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, alpha, beta, A_rows, A_cols, B_cols);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for GEMM (DataType: %s): %f ms\n", typeid(T).name(), milliseconds);

    // Copy the result back to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template void gemm<float>(const float *h_A, const float *h_B, float *h_C, float alpha, float beta, int A_rows, int A_cols, int B_cols);
template void gemm<double>(const double *h_A, const double *h_B, double *h_C, double alpha, double beta, int A_rows, int A_cols, int B_cols);