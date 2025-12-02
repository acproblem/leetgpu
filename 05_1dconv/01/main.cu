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


__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
    int input_size, int kernel_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float tmp = 0.0f;
    if (tid <= input_size - kernel_size) {
        for (int j = 0; j < kernel_size; j++) {
            tmp += input[tid + j] * kernel[j];
        }
        output[tid] = tmp;
    }
}


void convolution_1d_kernel_cpu(const float* input, const float* kernel, float* output,
    int input_size, int kernel_size) {
    for (int i = 0; i <= input_size - kernel_size; ++i) {
        float tmp = 0.0f;
        for (int j = 0; j < kernel_size; ++j) {
            tmp += input[i + j] * kernel[j];
        }
        output[i] = tmp;
    }
}



int main(int argc, char *argv[]) {
    int input_size = 1 << 28;  // 1 GB
    int kernel_size = 11;
    if (argc >= 2) input_size = std::atoi(argv[1]);
    if (argc >= 3) kernel_size = std::atoi(argv[2]);

    bool enable_check = false;
    if (argc >= 3) enable_check = true;

    std::cout << "input_size: " << input_size << ", kernel_size: " << kernel_size << std::endl;

    // alloc host memory
    float *h_input = new float[input_size];
    float *h_kernel = new float[kernel_size];
    float *h_output = new float[input_size - kernel_size + 1];

    // init host data
    for (int i = 0; i < input_size; ++i) {
        h_input[i] = i % 7;
    }
    for (int i = 0; i < kernel_size; ++i) {
        h_kernel[i] = i % 7;
    }
    for (int i = 0; i < input_size - kernel_size + 1; i++) {
        h_output[i] = 0.0f;
    }
    
    // alloc device memory
    float *d_input = nullptr, *d_kernel = nullptr, *d_output = nullptr;
    CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));
    CHECK(cudaMalloc(&d_output, (input_size - kernel_size + 1) * sizeof(float)));

    // copy from host to device
    CHECK(cudaMemcpy(d_input, h_input, input_size* sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // run and timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    dim3 block(128, 1);
    dim3 grid((input_size - kernel_size + block.x) / block.x, 1);  // (input_size - kernel_size + 1 + block.x - 1) / block.x
    std::cout << "block: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    std::cout << "grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    
    warmup<<<grid, block>>>();

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i)
        convolution_1d_kernel<<<grid, block>>>(d_input, d_kernel, d_output, input_size, kernel_size);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // copy from device to host
    CHECK(cudaMemcpy(h_output, d_output, (input_size - kernel_size + 1) * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "kernel time(ms): " << milliseconds / 10 << std::endl;

    // check
    if (enable_check) {
        float *output_cpu = new float[input_size - kernel_size + 1];
        convolution_1d_kernel_cpu(h_input, h_kernel, output_cpu, input_size, kernel_size);
        bool flag = true;
        for (int i = 0; i < input_size - kernel_size + 1; ++i) {
            if (output_cpu[i] != h_output[i]) {
                flag = false;
                std::cout << "i = " << i << ", output_cpu[i] = " << output_cpu[i] << ", h_output[i] = " << h_output[i] << std::endl;
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
    delete[] h_kernel;
    delete[] h_output;
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_kernel));
    CHECK(cudaFree(d_output));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}