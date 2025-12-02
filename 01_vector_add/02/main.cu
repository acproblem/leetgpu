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

#define TILE_SIZE 4

// 展开
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x * TILE_SIZE + threadIdx.x;
    for (int i = 0; i < TILE_SIZE; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < N) {
            C[idx] = A[idx] + B[idx];
        }
    }
}


int main(int argc, char *argv[]) {
    int N = 1 << 28;  // 256 M data
    int block_size_x = 256;
    if (argc >= 2) {
        N = std::atoi(argv[1]);
    } 
    if (argc >= 3) {
        block_size_x = std::atoi(argv[2]);
    }

    std::cout << "N: " << N << std::endl;

    // alloc host memory
    int nBytes = N * sizeof(float);
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // init host data
    for (int i = 0; i < N; ++i) {
        h_A[i] = h_B[i] = i % 7;
    }
    
    // alloc device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK(cudaMalloc(&d_A, nBytes));
    CHECK(cudaMalloc(&d_B, nBytes));
    CHECK(cudaMalloc(&d_C, nBytes));

    // copy from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    
    // run and timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    dim3 block(block_size_x, 1);
    dim3 grid((N + block.x * TILE_SIZE - 1) / (block.x * TILE_SIZE), 1);
    std::cout << "block: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    std::cout << "grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    
    warmup<<<grid, block>>>();

    CHECK(cudaEventRecord(start));
    vector_add<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0.0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // copy from device to host
    CHECK(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost));
    
    std::cout << "kernel time(ms): " << milliseconds << std::endl;
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