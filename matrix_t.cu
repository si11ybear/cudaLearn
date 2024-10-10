#include <stdio.h>

// CUDA Kernel function to transpose a matrix
__global__ void T(int *a, int *c, int width, int height) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int index = row * width + col;
    int index_t = row + col * height;
    if (row < height && col < width) {
        c[index_t] = a[index] ;
        // printf("%d,%d\n", index ,index_t);
    }
}

int main() {
    int width = 150;
    int height = 120; // Size of arrays
    int n = width * height;
    int size = n * sizeof(int);

    // Host copies of a, b, c
    int *h_a, *h_c;

    // Device copies of a, b, c
    int *d_a, *d_c;

    // Allocate space for host copies of a, b, c and setup input values
    h_a = (int *)malloc(size);
    h_c = (int *)malloc(size);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            h_a[row * width + col] = row + 2 * col;
        }
    }

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on the GPU
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    T<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c, width, height);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print out some results
    // Print result
    int h = 20 < height? 20:height;
    int w = 20 < width? 20:width;
    printf("Matrix A:\n");
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            printf("%d ", h_a[row * width + col]);
        }
        printf("\n");
    }

    printf("\nResult (Matrix A.T):\n");
    for (int row = 0; row < w; row++) {
        for (int col = 0; col < h; col++) {
            printf("%d ", h_c[row * height + col]);
        }
        printf("\n");
    }

    // Cleanup
    free(h_a);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_c);

    return 0;
}
