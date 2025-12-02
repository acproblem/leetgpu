#include <iostream>
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


#define BLOCK_SIZE_X 4
#define BLOCK_SIZE_Y 64
#define TILE_SIZE_X 2
#define TILE_SIZE_Y 2

// rows 和 cols 代表 input 矩阵的行与列
__global__ void transpose(const float *input, float *output, int rows, int cols) {
    int i = blockIdx.y * blockDim.y * TILE_SIZE_Y + threadIdx.y;
    int j = blockIdx.x * blockDim.x * TILE_SIZE_X + threadIdx.x;

    for (int t1 = 0; t1 < TILE_SIZE_Y; t1++) {
        for (int t2 = 0; t2 < TILE_SIZE_X; t2++) {
            if (i + t1 * blockDim.y < cols && j + t2 * blockDim.x < rows) {
                output[(i + t1 * blockDim.y) * rows + (j + t2 * blockDim.x)] = 
                    input[(j + t2 * blockDim.x) * cols + (i + t1 * blockDim.y)];
            }
        }
    }
}


// rows 和 cols 代表 input 矩阵的行与列
void transpose_cpu(const float *input, float *output, int rows, int cols) {
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            output[i * rows + j] = input[j * cols + i];
        }
    }
}


int main(int argc, char *argv[]) {
    int M = 16384;
    int N = 16384;
    if (argc >= 2) M = std::atoi(argv[1]);
    if (argc >= 3) N = std::atoi(argv[2]);

    bool enable_check = false;
    if (argc >= 4) enable_check = true;

    std::cout << "M: " << M << ", N: " << N << std::endl;

    // alloc host memory
    float *h_input = new float[M * N];
    float *h_output = new float[M * N];

    // init host data
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = i % 7;
    }
    for (int i = 0; i < M * N; ++i) {
        h_output[i] = i % 7;
    }
    
    // alloc device memory
    float *d_input = nullptr, *d_output = nullptr;
    CHECK(cudaMalloc(&d_input, M * N * sizeof(float)));
    CHECK(cudaMalloc(&d_output, M * N * sizeof(float)));

    // copy from host to device
    CHECK(cudaMemcpy(d_input, h_output, M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // run and timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((M + (block.x * TILE_SIZE_X) - 1) / (block.x * TILE_SIZE_X),
                (N + (block.y * TILE_SIZE_Y) - 1) / (block.y * TILE_SIZE_Y));
    std::cout << "block: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    std::cout << "grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    
    warmup<<<grid, block>>>();

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i)
        transpose<<<grid, block>>>(d_input, d_output, M, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // copy from device to host
    CHECK(cudaMemcpy(h_output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "kernel time(ms): " << milliseconds / 10 << std::endl;

    // check
    if (enable_check) {
        float *output_cpu = new float[M * N * sizeof(float)];
        transpose_cpu(h_input, output_cpu, M, N);
        bool flag = true;
        for (int i = 0; i < M * N; ++i) {
            if (output_cpu[i] != h_output[i]) {
                flag = false;
                std::cout << "i = " << i << ", C_cpu[i] = " << output_cpu[i] << ", h_C[i] = " << h_output[i] << std::endl;
                break;
            }
        }
        if (flag)
            std::cout << "Successful!" << std::endl;
        else
            std::cout << "Failed!" << std::endl;
        delete[] output_cpu;
    }

    // free resources
    delete[] h_input;
    delete[] h_output;
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}