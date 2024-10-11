#ifndef GEMV_H
#define GEMV_H

template <typename T>
void gemv_kernel(T *A, T *x, T *y, T alpha, T beta, int rows, int cols);

#endif // GEMV_H