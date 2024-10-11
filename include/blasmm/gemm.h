#ifndef GEMM_H
#define GEMM_H

template <typename T>
void gemm_kernel(T *A, T *x, T *y, T alpha, T beta, int A_rows, int A_cols, int B_cols);

#endif // GEMM_H