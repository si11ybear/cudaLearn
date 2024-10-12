#include <stdio.h>
#include <typeinfo>

// CUDA Kernel function to add elements of two arrays
// Templated CUDA kernel for GEMV: y = alpha * x + y
template <typename T>
__global__ void add(T *x, T alpha, T *y, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        y[index] = alpha * x[index] + y[index];
    }
}

template <typename T>
void vector_add(const T* h_a, const T alpha, T* h_b, int n) {
    // int n = 1000; // Size of arrays
    int size = n * sizeof(int);

    // Host copies of a, b, c
    // T *h_a, *h_b, *h_c;

    // Device copies of a, b, c
    T *d_a, *d_b;

    // // Allocate space for host copies of a, b, c and setup input values
    // h_a = (T *)malloc(size);
    // h_b = (T *)malloc(size);
    // h_c = (T *)malloc(size);

    // for (int i = 0; i < n; i++) {
    //     h_a[i] = i;
    //     h_b[i] = i * 2;
    // }

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on the GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, alpha, d_b, n);

    // Copy result back to host
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    // // Print out some results
    // for (int i = 0; i < 10; i++) {
    //     printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    // }

    // // Cleanup
    // free(h_a);
    // free(h_b);
    // free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);

    return;
}

template void vector_add<float>(const float* h_a, float alpha, float* h_b, int n);
template void vector_add<double>(const double* h_a, double alpha, double* h_b, int n);
