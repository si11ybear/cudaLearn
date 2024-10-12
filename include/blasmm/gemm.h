#ifndef GEMM_H
#define GEMM_H

template <typename T>
void gemm(const T *h_A, const T *h_B, T *h_C, T alpha, T beta, int A_rows, int A_cols, int B_cols);

#endif // GEMM_H