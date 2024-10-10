#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel function to add elements of two arrays
__global__ void matrix_add_ker(int *a, int *b, int *c, int width, int height) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int index = row * width + col;
    if (row < height && col < width) {
        c[index] = a[index] + b[index];
    }
}

void matrix_add(const int* h_a, const int* h_b, int* h_c, int width, int height) {
    // int width = 200;
    // int height = 300; // Size of arrays
    int n = width * height;
    int size = n * sizeof(int);

    // Device copies of a, b, c
    int *d_a, *d_b, *d_c;

    // Allocate space for host copies of a, b, c and setup input values
    // h_a = (int *)malloc(size);
    // h_b = (int *)malloc(size);
    // h_c = (int *)malloc(size);

    // for (int row = 0; row < height; row++) {
    //     for (int col = 0; col < width; col++) {
    //         h_a[row * width + col] = row + col;
    //         h_b[row * width + col] = row - col;
    //     }
    // }

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on the GPU
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
   
    matrix_add_ker<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width, height);

    // Time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Performance metrics
    // 1. Time
    printf("Time for matrix addition: %f ms\n", milliseconds);

    // 2. FLOPS (Matrix Add: 1 addition per element)
    int flops = width * height;  // Each element involves 1 addition
    float flops_per_second = (flops / (milliseconds / 1000.0f)) / 1e9;  // Convert to GFLOPS
    printf("Performance: %f GFLOPS\n", flops_per_second);

    // 3. Bandwidth (Memory bound calculation)
    // We are reading two matrices (a, b) and writing one matrix (c)
    float bandwidth = (3.0f * size / (milliseconds / 1000.0f)) / (1 << 30);  // GB/s
    printf("Memory Bandwidth: %f GB/s\n", bandwidth);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print out some results
    // Print result
    // int h = 20 < height? 20:height;
    // int w = 20 < width? 20:width;
    // printf("%d,%d", h, w);
    // printf("Matrix A:\n");
    // for (int row = 0; row < h; row++) {
    //     for (int col = 0; col < w; col++) {
    //         printf("%d ", h_a[row * width + col]);
    //     }
    //     printf("\n");
    // }

    // printf("\nMatrix B:\n");
    // for (int row = 0; row < h; row++) {
    //     for (int col = 0; col < w; col++) {
    //         printf("%d ", h_b[row * width + col]);
    //     }
    //     printf("\n");
    // }

    // printf("\nResult (Matrix C = A + B):\n");
    // for (int row = 0; row < h; row++) {
    //     for (int col = 0; col < w; col++) {
    //         printf("%d ", h_c[row * width + col]);
    //     }
    //     printf("\n");
    // }

    // Cleanup
    // free(h_a);
    // free(h_b);
    // free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return;
}
