#include <iostream>
#include <string>
#include <cuda_runtime.h>


#define CHECK(call)                                                     \
do {                                                                    \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                  << ", code=" << err << " (" << cudaGetErrorString(err) << ")\n"; \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

__global__ void warmup() {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
}


#define BM 32
#define BN 32

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    float tmp = 0.0f;
    if (row < M && col < N) {
        for (int k = 0; k < K; ++k) {
            tmp += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = tmp;
    }
}


void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float tmp = 0.0f;
            for (int k = 0; k < K; ++k) {
                tmp += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = tmp;
        }
    }
}



int main(int argc, char *argv[]) {
    int M = 8192;
    int N = 8192;
    int K = 8192;
    if (argc >= 2) M = std::atoi(argv[1]);
    if (argc >= 3) N = std::atoi(argv[2]);
    if (argc >= 4) K = std::atoi(argv[3]);

    bool enable_check = false;
    if (argc >= 5) enable_check = true;

    std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

    // alloc host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // init host data
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = i % 7;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = i % 7;
    }
    
    // alloc device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // copy from host to device
    CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // run and timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    std::cout << "block: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    std::cout << "grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    
    warmup<<<grid, block>>>();

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i)
        matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // copy from device to host
    CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "kernel time(ms): " << milliseconds / 10 << std::endl;

    // check
    if (enable_check) {
        float *C_cpu = new float[M * N];
        matmul_cpu(h_A, h_B, C_cpu, M, N, K);
        bool flag = true;
        for (int i = 0; i < M * N; ++i) {
            if (C_cpu[i] != h_C[i]) {
                flag = false;
                std::cout << "i = " << i << ", C_cpu[i] = " << C_cpu[i] << ", h_C[i] = " << h_C[i] << std::endl;
                break;
            }
        }
        if (flag)
            std::cout << "Successful!" << std::endl;
        else
            std::cout << "Failed!" << std::endl;
        delete[] C_cpu;
    }

    // free resources
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}