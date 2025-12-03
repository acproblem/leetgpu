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

#define BLOCK_SIZE 256  // 2 的整数次方且 >= 32

// 交错规约 + 完全展开
__global__ void reducation(float* input, float* output, int N) {
    int tid = BLOCK_SIZE * blockIdx.x * 2 + threadIdx.x;

    // 完全展开
    if (BLOCK_SIZE >= 1024 && threadIdx.x < 1024 && tid + 1024 < N) {
        input[tid] += input[tid + 1024];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 512 && threadIdx.x < 512 && tid + 512 < N) {
        input[tid] += input[tid + 512];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256 && threadIdx.x < 256 && tid + 256 < N) {
        input[tid] += input[tid + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128 && threadIdx.x < 128 && tid + 128 < N) {
        input[tid] += input[tid + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 64 && threadIdx.x < 64 && tid + 64 < N) {
        input[tid] += input[tid + 64];
        __syncthreads();
    }

    // warp 级展开
    if (threadIdx.x < 32) {
        volatile float *vsdata = &(input[BLOCK_SIZE * blockIdx.x * 2]);
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 32];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 16];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 8];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 4];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 2];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 1];
    }

    if (threadIdx.x == 0)
        output[blockIdx.x] = input[tid];
}


float reducation_cpu(float* a, int N) {
    if (N < 16384) {
        float res = 0.0f;
        for (int i = 0; i < N; i++)
            res += a[i];
        return res;
    } else {
        return reducation_cpu(a, N / 2) + reducation_cpu(a + N / 2, N - N / 2);
    }
}



int main(int argc, char *argv[]) {
    int N = 1 << 28;  // 1 GB
    if (argc >= 2) N = std::atoi(argv[1]);

    bool enable_check = false;
    if (argc >= 3) enable_check = true;

    std::cout << "N: " << N << std::endl;

    int output_size = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    // alloc host memory
    float *h_input = new float[N];
    float *h_output = new float[output_size];

    // init host data
    for (int i = 0; i < N; ++i) {
        h_input[i] = i % 7;
    }
    for (int i = 0; i < output_size; ++i) {
        h_output[i] = 0.0f;
    }
    
    // alloc device memory
    float *d_input = nullptr, *d_output = nullptr;
    CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));

    // copy from host to device
    CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // run and timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    dim3 block(BLOCK_SIZE, 1);
    dim3 grid(output_size, 1);
    std::cout << "block: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    std::cout << "grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    
    warmup<<<grid, block>>>();

    CHECK(cudaEventRecord(start));
    reducation<<<grid, block>>>(d_input, d_output, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // copy from device to host
    CHECK(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "kernel time(ms): " << milliseconds << std::endl;

    // check
    if (enable_check) {
        float res_cpu = reducation_cpu(h_input, N);
        float res_gpu = reducation_cpu(h_output, output_size);
        std::cout << "CPU result: " << res_cpu << ", GPU result: " << res_gpu << std::endl;
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