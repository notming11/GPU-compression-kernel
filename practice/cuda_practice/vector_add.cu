#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <chrono>
#include <iostream>

__global__ void vecAdd(float* A, float* B, float* C, int l){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < l){
        C[index] = A[index] + B[index];
    }
}

void vec_rand(float* A, int l){
    for(int i = 0; i < l; i++){
        A[i] = rand() / rand();
    }
}

int main(){
    int l;
    std::cin >> l;

    float* vec_A = nullptr;
    float* vec_B = 0;
    float* vec_C = 0;
    
    cudaMallocHost(&vec_A, sizeof(float) * l); 
    cudaMallocHost(&vec_B, sizeof(float) * l); 
    cudaMallocHost(&vec_C, sizeof(float) * l); 

    
    vec_rand(vec_A, l);
    vec_rand(vec_B, l);
    
    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < l; i++){
        vec_C[i] = vec_B[i] + vec_A[i];
    }
    auto end = std::chrono::steady_clock::now();
    
    std::cout << "VecAdd on CPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // initialize device memory
    float* A_gpu = 0;
    float* B_gpu = 0;
    float* C_gpu = 0;

    cudaMalloc(&A_gpu, sizeof(float) * l);
    cudaMalloc(&B_gpu, sizeof(float) * l);
    cudaMalloc(&C_gpu, sizeof(float) * l);

    cudaMemcpy(A_gpu, vec_A, sizeof(float)*l, cudaMemcpyDefault);
    cudaMemcpy(B_gpu, vec_B, sizeof(float)*l, cudaMemcpyDefault);
    cudaMemset(C_gpu, 0, sizeof(float)*l);

    int thread = 32, block = (l + thread-1)/thread;
    start = std::chrono::steady_clock::now();
    vecAdd<<<block, thread>>>(A_gpu, B_gpu, C_gpu, l);
    end = std::chrono::steady_clock::now();
    std::cout << "VecAdd on GPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    float* gpu_result = 0;
    cudaMallocHost(&gpu_result, sizeof(float)*l);
    cudaMemcpy(gpu_result, C_gpu, sizeof(float)*l, cudaMemcpyDefault);

    bool b = 0;
    for(int i = 0; i < l; i++){
        if (gpu_result[i] - vec_C[i] > 1e-4){
            b = 1;
        }
    }

    if (b){
        std::cout << "Mismatched" << std::endl;
    } else {
        std::cout << "Matched" << std::endl;
    }

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFreeHost(vec_A);
    cudaFreeHost(vec_B);
    cudaFreeHost(vec_C);
    cudaFreeHost(gpu_result);
    return 0;
}