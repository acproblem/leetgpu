#include <iostream>
#include <string>
#include <cmath>
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

#define BLOCK_SIZE1 128
#define BLOCK_SIZE2 64

// 计算分母
__global__ void softmax_kernel1(float* input, float* global_sum, int N) {
    int tid = BLOCK_SIZE1 * blockIdx.x * 2 + threadIdx.x;

    // ----- 计算每个元素的指数并保存在共享内存 -----
    __shared__ float sdata[BLOCK_SIZE1 * 2];  // 块内规约数据
    if (tid < N)
        sdata[threadIdx.x] = exp(input[tid]);
    else
        sdata[threadIdx.x] = 0.0f;
    if (tid + BLOCK_SIZE1 < N)
        sdata[threadIdx.x + BLOCK_SIZE1] = exp(input[tid + BLOCK_SIZE1]);
    else
        sdata[threadIdx.x + BLOCK_SIZE1] = 0.0f;
    __syncthreads();

    // ----- 规约，完全展开 -----
    if (BLOCK_SIZE1 >= 1024 && threadIdx.x < 1024) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 1024];
        __syncthreads();
    }
    if (BLOCK_SIZE1 >= 512 && threadIdx.x < 512) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 512];
        __syncthreads();
    }
    if (BLOCK_SIZE1 >= 256 && threadIdx.x < 256) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE1 >= 128 && threadIdx.x < 128) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE1 >= 64 && threadIdx.x < 64) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 64];
        __syncthreads();
    }
    // warp 级展开
    if (threadIdx.x < 32) {
        volatile float *vsdata = sdata;
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 32];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 16];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 8];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 4];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 2];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 1];
    }
    if (threadIdx.x == 0)
        atomicAdd(global_sum, sdata[0]);
}

// 计算指数并相除
__global__ void softmax_kernel2(float* input, float* output, float* global_sum, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // ----- 计算 output -----
    if (tid < N)
        output[tid] = exp(input[tid]) / (*global_sum);
}


void softmax_gpu(float* input, float* output, int N) {
    float* global_sum;
    cudaMalloc(&global_sum, sizeof(float));
    cudaMemset(global_sum, 0, sizeof(float));
    dim3 block1(BLOCK_SIZE1, 1);
    dim3 grid1((N + 2 * BLOCK_SIZE1 - 1) / (2 * BLOCK_SIZE1), 1);
    softmax_kernel1<<<grid1, block1>>>(input, global_sum, N);

    dim3 block2(BLOCK_SIZE2, 1);
    dim3 grid2((N + BLOCK_SIZE2 - 1) / (BLOCK_SIZE2), 1);
    softmax_kernel2<<<grid2, block2>>>(input, output, global_sum, N);
}


float sum_exp_cpu(float *a, int N) {
    if (N < 1024) {
        float res = 0.0f;
        for (int i = 0; i < N; i++) {
            res += exp(a[i]);
        }
        return res;
    } else {
        return sum_exp_cpu(a, N / 2) + sum_exp_cpu(a + N / 2, N - N / 2);
    }
}

void softmax_cpu(float* a, float *b, int N) {
    float t = sum_exp_cpu(a, N);
    for (int i = 0; i < N; i++) {
        b[i] = exp(a[i]) / t;
    }
}


int main(int argc, char *argv[]) {
    int N = 1 << 28;  // 1 GB
    if (argc >= 2) N = std::atoi(argv[1]);

    bool enable_check = false;
    if (argc >= 3) enable_check = true;

    std::cout << "N: " << N << std::endl;

    // alloc host memory
    float *h_input = new float[N];
    float *h_output = new float[N];

    // init host data
    for (int i = 0; i < N; ++i) {
        h_input[i] = i % 7;
    }
    for (int i = 0; i < N; ++i) {
        h_output[i] = 0.0f;
    }
    
    // alloc device memory
    float *d_input = nullptr, *d_output = nullptr;
    CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    // copy from host to device
    CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // run and timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    warmup<<<1024, 64>>>();

    CHECK(cudaEventRecord(start));
    softmax_gpu(d_input, d_output, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // copy from device to host
    CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "kernel time(ms): " << milliseconds << std::endl;

    // check
    if (enable_check) {
        float *output_cpu = new float[N];
        softmax_cpu(h_input, output_cpu, N);
        for (int i = 0; i < N; i++) {
            if (fabs(h_output[i] - output_cpu[i]) > 1e-6) {
                std::cout << "i = " << i << ", h_output[i] = " << h_output[i] << ", output_cpu[i] = " << output_cpu[i] << std::endl;
                return -1;
            }
        }
        std::cout << "Check pass!" << std::endl;
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