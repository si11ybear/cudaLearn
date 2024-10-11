#include <iostream>
#include "blasmm/matrix_add.h"
#include "blasmm/matrix_t.h"

// 初始化矩阵数据的函数
void init_matrix(int* matrix, int width, int height, bool identity = false) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (identity) {
                matrix[row * width + col] = (row == col) ? 1 : 0; // 单位矩阵
            } else {
                matrix[row * width + col] = row + col; // 简单初始化
            }
        }
    }
}

// 打印矩阵的函数
void print_matrix(const int* matrix, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            std::cout << matrix[row * width + col] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // 矩阵的尺寸
    int width = 4;
    int height = 3;

    // 分配并初始化主机上的矩阵
    int* h_a = new int[width * height];
    int* h_b = new int[width * height];
    int* h_c = new int[width * height];  // 用于存储加法结果
    int* h_t = new int[width * height];  // 用于存储转置结果

    // 初始化矩阵 A 和 B
    init_matrix(h_a, width, height);
    init_matrix(h_b, width, height, true); // B 初始化为单位矩阵

    // 输出原始矩阵 A 和 B
    std::cout << "Matrix A:" << std::endl;
    print_matrix(h_a, width, height);

    std::cout << "Matrix B (Identity):" << std::endl;
    print_matrix(h_b, width, height);

    // 调用 CUDA 矩阵加法函数
    matrix_add(h_a, h_b, h_c, width, height);

    // 输出加法结果
    std::cout << "Matrix C (A + B):" << std::endl;
    print_matrix(h_c, width, height);

    // 调用 CUDA 矩阵转置函数
    matrix_T(h_a, h_t, width, height);

    // 输出转置结果
    std::cout << "Matrix A Td:" << std::endl;
    print_matrix(h_t, height, width);

    // 释放主机上的矩阵内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_t;

    return 0;
}
