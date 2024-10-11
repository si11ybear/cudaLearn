#include <iostream>
#include <cstring>
#include <chrono>
#include <ctime>
#include "blas.h"  // 统一头文件

// 打印帮助信息
void print_help() {
    std::cout << "Usage: ./main_exec --op <operation> [--size <size>] [--width <width>] [--height <height>]\n";
    std::cout << "Operations:\n";
    std::cout << "  vector_add: Test vector addition\n";
    std::cout << "  matrix_add: Test matrix addition\n";
    std::cout << "  matrix_transpose: Test matrix transpose\n";
}

// 用于 CPU 的向量加法
void vector_add_cpu(const int* a, const int* b, int* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// 用于 CPU 的矩阵加法，用于对比 GPU 结果
void matrix_add_cpu(const int* a, const int* b, int* c, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int index = row * width + col;
            c[index] = a[index] + b[index];
        }
    }
}

// 用于 CPU 的矩阵转置，用于对比 GPU 结果
void matrix_transpose_cpu(const int* input, int* output, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int input_index = row * width + col;
            int output_index = col * height + row;
            output[output_index] = input[input_index];
        }
    }
}

// 随机生成矩阵
void generate_random_matrix(int* mat, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        mat[i] = rand() % 100;  // 随机生成 0-99 范围的整数
    }
}

// 随机生成向量
void generate_random_vector(int* vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = rand() % 100;
    }
}

// 测试向量加法的性能与正确性
void test_vector_add(int n) {
    int* h_a = (int*)malloc(n * sizeof(int));
    int* h_b = (int*)malloc(n * sizeof(int));
    int* h_c_cpu = (int*)malloc(n * sizeof(int));
    int* h_c_gpu = (int*)malloc(n * sizeof(int));

    generate_random_vector(h_a, n);
    generate_random_vector(h_b, n);

    // CPU 计算向量加法
    vector_add_cpu(h_a, h_b, h_c_cpu, n);

    // 测试 GPU 向量加法性能
    auto start = std::chrono::high_resolution_clock::now();
    vector_add(h_a, h_b, h_c_gpu, n);  // GPU 向量加法
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "GPU Vector Addition took: " << elapsed.count() << " seconds.\n";

    // 检查结果正确性
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (h_c_cpu[i] != h_c_gpu[i]) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << h_c_cpu[i] << ", GPU=" << h_c_gpu[i] << "\n";
            break;
        }
    }

    if (correct) {
        std::cout << "Vector addition result is correct.\n";
    } else {
        std::cout << "Vector addition result is incorrect.\n";
    }

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
}

// 测试矩阵加法的性能与正确性
void test_matrix_add(int width, int height) {
    int size = width * height;
    
    // 分配矩阵内存
    int* h_a = (int*)malloc(size * sizeof(int));
    int* h_b = (int*)malloc(size * sizeof(int));
    int* h_c_cpu = (int*)malloc(size * sizeof(int));  // CPU 结果
    int* h_c_gpu = (int*)malloc(size * sizeof(int));  // GPU 结果

    generate_random_matrix(h_a, width, height);
    generate_random_matrix(h_b, width, height);

    // CPU 测试矩阵加法正确性
    matrix_add_cpu(h_a, h_b, h_c_cpu, width, height);

    // 测试 GPU 性能
    auto start = std::chrono::high_resolution_clock::now();
    matrix_add(h_a, h_b, h_c_gpu, width, height);  // GPU 矩阵加法
    auto end = std::chrono::high_resolution_clock::now();
    
    // 计算执行时间
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "GPU Matrix Addition took: " << elapsed.count() << " seconds.\n";

    // 检查 GPU 结果的正确性
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (h_c_cpu[i] != h_c_gpu[i]) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << h_c_cpu[i] << ", GPU=" << h_c_gpu[i] << "\n";
            break;
        }
    }

    if (correct) {
        std::cout << "Matrix addition result is correct.\n";
    } else {
        std::cout << "Matrix addition result is incorrect.\n";
    }

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
}

// 测试矩阵转置的性能与正确性
void test_matrix_transpose(int width, int height) {
    int size = width * height;
    
    // 分配矩阵内存
    int* h_input = (int*)malloc(size * sizeof(int));
    int* h_output_cpu = (int*)malloc(size * sizeof(int));  // CPU 结果
    int* h_output_gpu = (int*)malloc(size * sizeof(int));  // GPU 结果

    generate_random_matrix(h_input, width, height);

    // CPU 测试矩阵转置
    matrix_transpose_cpu(h_input, h_output_cpu, width, height);

    // 测试 GPU 性能
    auto start = std::chrono::high_resolution_clock::now();
    matrix_T(h_input, h_output_gpu, width, height);  // GPU 矩阵转置
    auto end = std::chrono::high_resolution_clock::now();
    
    // 计算执行时间
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "GPU Matrix Transpose took: " << elapsed.count() << " seconds.\n";

    // 检查 GPU 结果的正确性
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (h_output_cpu[i] != h_output_gpu[i]) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << h_output_cpu[i] << ", GPU=" << h_output_gpu[i] << "\n";
            break;
        }
    }

    if (correct) {
        std::cout << "Matrix transpose result is correct.\n";
    } else {
        std::cout << "Matrix transpose result is incorrect.\n";
    }

    // 释放内存
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
}

// 解析命令行参数并运行相应的测试
int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_help();
        return 1;
    }

    std::string op = "";
    int size = 0, width = 0, height = 0;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--op") == 0) {
            op = argv[++i];  // 获取操作类型
        } else if (strcmp(argv[i], "--size") == 0) {
            size = atoi(argv[++i]);  // 获取向量大小
        } else if (strcmp(argv[i], "--width") == 0) {
            width = atoi(argv[++i]);  // 获取矩阵宽度
        } else if (strcmp(argv[i], "--height") == 0) {
            height = atoi(argv[++i]);  // 获取矩阵高度
        }
    }

    // 根据命令行参数选择测试
    if (op == "vector_add") {
        if (size == 0) {
            std::cout << "Error: You must provide --size for vector_add operation.\n";
            return 1;
        }
        std::cout << "Testing vector addition...\n";
        test_vector_add(size);  // 调用向量加法测试函数
    } else if (op == "matrix_add") {
        if (width == 0 || height == 0) {
            std::cout << "Error: You must provide --width and --height for matrix_add operation.\n";
            return 1;
        }
        std::cout << "Testing matrix addition...\n";
        test_matrix_add(width, height);  // 调用矩阵加法测试函数
    } else if (op == "matrix_transpose") {
        if (width == 0 || height == 0) {
            std::cout << "Error: You must provide --width and --height for matrix_transpose operation.\n";
            return 1;
        }
        std::cout << "Testing matrix transpose...\n";
        test_matrix_transpose(width, height);  // 调用矩阵转置测试函数
    } else {
        print_help();
        return 1;
    }

    return 0;
}
