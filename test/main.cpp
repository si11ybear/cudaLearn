#include <iostream>
#include "cublas_v2.h"

#define  N 10
int main() {
    float a[N], b[N], c[N];
    float *dev_a, *dev_b, *dev_c;
    for(int i=0; i<N; ++i) // 为数组a、b赋值
    {
        float tmp = 1.0 * i;
        a[i] = tmp;
        b[i] = tmp * tmp;
    }

    cublasHandle_t handle;  // 申明句柄
    cublasCreate_v2(&handle); // 创建句柄
    cudaMalloc((void**)&dev_a, sizeof(float) * N);
    cudaMalloc((void**)&dev_b, sizeof(float) * N);
//
    float alpha = 1.0;
    cublasSetVector(N, sizeof(float), a, 1, dev_a, 1); // H2D host to device
    cublasSetVector(N, sizeof(float), b, 1, dev_b, 1);
    cublasSaxpy_v2(handle, N, &alpha, dev_a, 1, dev_b, 1); //实现向量+
    cublasGetVector(N, sizeof(float), dev_b, 1, c, 1); // D2H
    cudaFree(dev_a);
    cudaFree(dev_b);
    cublasDestroy(handle); // 销毁句柄

    for(int i=0; i<N; ++i)
    {
        printf("%f + %f * %f = %f \n", a[i], b[i],b[i], c[i]);
    }
    std::cout << "Hello, World!" << std::endl;
    return 0;
}

