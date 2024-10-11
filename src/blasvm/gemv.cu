#include <stdio.h>
#include <cuda_runtime.h>
#include <typeinfo>

// Templated CUDA kernel for GEMV: y = alpha * A * x + beta * y
template <typename T>
__global__ void gemv_kernel(T *A, T *x, T *y, T alpha, T beta, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        T result = 0;
        for (int col = 0; col < cols; col++) {
            result += A[row * cols + col] * x[col];
        }
        y[row] = alpha * result + beta * y[row];
    }
}

// Templated GEMV function with FLOPS and Bandwidth calculation
template <typename T>
void gemv(const T *h_A, const T *h_x, T *h_y, T alpha, T beta, int rows, int cols) {
    // Calculate sizes
    int size_A = rows * cols * sizeof(T);
    int size_x = cols * sizeof(T);
    int size_y = rows * sizeof(T);

    // Device memory pointers
    T *d_A, *d_x, *d_y;

    // Allocate memory on the device (GPU)
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_x, size_x);
    cudaMalloc((void **)&d_y, size_y);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size_y, cudaMemcpyHostToDevice);

    // Define thread block and grid size
    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start, 0);

    // Launch the kernel
    gemv_kernel<<<gridSize, blockSize>>>(d_A, d_x, d_y, alpha, beta, rows, cols);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate FLOPS
    size_t flops = 2 * static_cast<size_t>(rows) * static_cast<size_t>(cols); // 2 operations (mul + add) per element
    double gflops = (flops / 1e9) / (milliseconds / 1000.0);  // GFLOPS = flops / 10^9

    // Calculate memory bandwidth
    size_t mem_access = (rows * cols + cols + rows) * sizeof(T);  // Total memory accessed
    double bandwidth = (mem_access / 1e9) / (milliseconds / 1000.0);  // Bandwidth in GB/s

    // Output the results
    printf("Time for GEMV (DataType: %s): %f ms\n", typeid(T).name(), milliseconds);
    printf("Performance: %f GFLOPS\n", gflops);
    printf("Memory Bandwidth: %f GB/s\n", bandwidth);

    // Copy the result back to host
    cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}