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

#define BLOCK_SIZE1 64
#define BLOCK_SIZE2 128  // 2 的整数次方
#define BLOCK_SIZE3 64


// 1. 计算指数
__global__ void softmax_kernel1(float* input, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N)
        input[tid] = exp(input[tid]);
}


// 2. 计算分母
__global__ void softmax_kernel2(float* input, float* global_sum, int N) {
    int tid = BLOCK_SIZE2 * blockIdx.x * 2 + threadIdx.x;

    __shared__ float sdata[BLOCK_SIZE2 * 2];

    // 全局内存数据读取到共享内存
    if (tid < N)
        sdata[threadIdx.x] = input[tid];
    else
        sdata[threadIdx.x] = 0.0f;
    if (tid + BLOCK_SIZE2 < N)
        sdata[threadIdx.x + BLOCK_SIZE2] = input[tid + BLOCK_SIZE2];
    else
        sdata[threadIdx.x + BLOCK_SIZE2] = 0.0f;
    __syncthreads();

    // 完全展开
    if (BLOCK_SIZE2 >= 1024 && threadIdx.x < 1024) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 1024];
        __syncthreads();
    }
    if (BLOCK_SIZE2 >= 512 && threadIdx.x < 512) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 512];
        __syncthreads();
    }
    if (BLOCK_SIZE2 >= 256 && threadIdx.x < 256) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE2 >= 128 && threadIdx.x < 128) {
        sdata[threadIdx.x] += sdata[threadIdx.x + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE2 >= 64 && threadIdx.x < 64) {
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

// 3. 相除
__global__ void softmax_kernel3(float* input, float* output, float* global_sum, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N)
        output[tid] = input[tid] / (*global_sum);
}


float sum_cpu(float* a, int N) {
    if (N < 1024) {
        float res = 0.0f;
        for (int i = 0; i < N; i++) {
            res += a[i];
        }
        return res;
    } else {
        return sum_cpu(a, N / 2) + sum_cpu(a + N / 2, N - N / 2);
    }
}

void softmax_gpu(float* input, float* output, int N) {
    dim3 block1(BLOCK_SIZE1, 1);
    dim3 grid1((N + block1.x - 1) / block1.x, 1);
    softmax_kernel1<<<grid1, block1>>>(input, N);

    int tmp_size = (N + 2 * BLOCK_SIZE2 - 1) / (2 * BLOCK_SIZE2);
    float* global_sum = nullptr;
    cudaMalloc(&global_sum, sizeof(float));
    cudaMemset(global_sum, 0, sizeof(float));
    dim3 block2(BLOCK_SIZE2, 1);
    dim3 grid2(tmp_size, 1);
    softmax_kernel2<<<grid2, block2>>>(input, global_sum, N);

    dim3 block3(BLOCK_SIZE3, 1);
    dim3 grid3((N + block3.x - 1) / block3.x, 1);
    softmax_kernel3<<<grid3, block3>>>(input, output, global_sum, N);
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