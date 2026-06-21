#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <cmath>

# define index(row, col, n) ((row) * (n) + (col))

void Mat_rand(float* A, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            A[i * n + j] = rand() / (rand()+1);
        }
    }
}

void mat_print(float* A, int n){
    for(int i = 0; i < n; i++){
        for(int j =0; j < n; j++){
            std::cout << A[index(i, j, n)] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

void match(float* A, float* B, int n){
    for(int i = 0; i < n; i++){
        for(int j= 0; j < n; j++){
            if (std::fabs(A[index(i, j, n)] - B[index(i, j, n)]) > 1e-4){
                std::cout << "Mismatch at: " << i << ' ' << j << std::endl;
                return;
            }
        }
    }
    std::cout << "Match" << std::endl;
}

__global__ void Mat_mult_coalesced(float* A, float* B, float* C, int n){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < n && col < n){
        float sum =0;
        for(int k = 0; k < n; k++){
            sum += A[index(row, k, n)] * B[index(k, col, n)];
        }
        C[index(row, col, n)] = sum;
    }
}

__global__ void Mat_mult_shared(float* A, float* B, float* C, int n){
    // global index
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    float sum = 0;
    int tile_cnt = (n + 15)/16;
    for(int i = 0; i < tile_cnt; i++){
        if (threadIdx.x + blockDim.x * i < n){
            tileA[threadIdx.y][threadIdx.x] = A[index(row, threadIdx.x + blockDim.x * i, n)];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }

        if (threadIdx.y + blockDim.y * i < n){
            tileB[threadIdx.y][threadIdx.x] = B[index(threadIdx.y + blockDim.y * i, col, n)];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for(int k = 0; k < 16; k++){
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (col < n && row < n){
        C[index(row, col, n)] = sum;
    }
}

int main(){
    int n;
    std::cin >> n;

    // Initialize Matrix on Host
    float* A = 0;
    float* B = 0;
    float* C = 0;
    cudaMallocHost(&A, sizeof(float) * n * n);
    cudaMallocHost(&B, sizeof(float) * n * n);
    cudaMallocHost(&C, sizeof(float) * n * n);

    Mat_rand(A, n);
    Mat_rand(B, n);

    // mat_print(A, n);
    // mat_print(B, n);
    
    // Setup for Kernel with 2D Block and 2D Grid 
    dim3 threadsPerBlock(16, 16);
    
    int Block_x = (n + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int Block_y = (n + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 Blocks(Block_x, Block_y);
    
    float* A_gpu = 0;
    float* B_gpu = 0;
    float* C_gpu = 0;
    
    cudaMalloc(&A_gpu, sizeof(float) * n * n);
    cudaMalloc(&B_gpu, sizeof(float) * n * n);
    cudaMalloc(&C_gpu, sizeof(float) * n * n);
    
    cudaMemcpy(A_gpu, A, sizeof(float) * n * n, cudaMemcpyDefault);
    cudaMemcpy(B_gpu, B, sizeof(float) * n * n, cudaMemcpyDefault);
    cudaMemset(C_gpu, 0, sizeof(float) * n * n);
    
    // Run Matrix Multiplication on GPU with Coalesced Memory transaction
    auto start = std::chrono::steady_clock::now();   
    Mat_mult_coalesced<<<Blocks, threadsPerBlock>>>(A_gpu, B_gpu, C_gpu, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();   
    std::cout << "Matrix mult for GPU with Coalesced Memory Transfer: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    
    // Copy result from Device to Host
    cudaMemcpy(C, C_gpu, sizeof(float) * n * n, cudaMemcpyDefault);
    // mat_print(C, n);
    
    // Run Matrix Multiplication on GPU with Shared Memory between threads in Blocks
    cudaMemset(C_gpu, 0, sizeof(float) * n * n);
    start = std::chrono::steady_clock::now();   
    Mat_mult_shared<<<Blocks, threadsPerBlock>>>(A_gpu, B_gpu, C_gpu, n);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();   
    std::cout << "Matrix mult for GPU with shared Memory: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    
    // Copy result from Device to Host and compare
    float* gpu_result = 0;
    cudaMallocHost(&gpu_result, sizeof(float) * n * n);
    cudaMemcpy(gpu_result, C_gpu, sizeof(float) * n * n, cudaMemcpyDefault);
    // mat_print(gpu_result, n);

    match(gpu_result, C, n);
}