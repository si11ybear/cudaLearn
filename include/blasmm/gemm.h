#ifndef GEMM_H
#define GEMM_H

void gemm(const float *h_A, const float *h_B, float *h_C, float alpha, float beta, int A_rows, int A_cols, int B_cols, int iter);

#endif // GEMM_H