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
            A[i * n + j] = (float)rand() / (float) RAND_MAX + 0.001f;
        }
    }
}

void Mat_rand_2_4(float* A, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j += 4){
            A[index(i, j, n)] = 0;
            A[index(i, j+1, n)] = 0;
            A[index(i, j+2, n)] = 0;
            A[index(i, j+3, n)] = 0;

            int pos1 = rand() % 4, pos2 = 0;
            while(pos2 == pos1){
                pos2 = rand() % 4;
            }

            A[index(i, j + pos1, n)] = (float)rand() / (float) RAND_MAX + 0.001f;
            A[index(i, j + pos2, n)] = (float)rand() / (float) RAND_MAX + 0.001f;
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
            if (std::fabs(A[index(i, j, n)] - B[index(i, j, n)]) > 1e-3){
                std::cout << "Mismatch at: " << i << ' ' << j << std::endl;
                return;
            }
        }
    }
    std::cout << "Match" << std::endl;
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
        if (row < n && threadIdx.x + blockDim.x * i < n){
            tileA[threadIdx.y][threadIdx.x] = A[index(row, threadIdx.x + blockDim.x * i, n)];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < n && threadIdx.y + blockDim.y * i < n){
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

__global__ void prune_2_4(float* A_sparse, float* compressed, short* metadata, int n){
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;

    int cnt = 0;
    for(int i = 0; i < 4; i++){
        if (idx + i < n * n && A_sparse[idx + i] != 0){
            compressed[idx/2 + cnt] = A_sparse[idx+i];
            metadata[idx/2 + cnt++] = i;
        }
    }
}

__global__ void SpMM(float* compressed, short* metadata, float* B, float* C, int n){
    // global index
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float tilecompressed[16][8];
    __shared__ short tilemetadata[16][8];
    __shared__ float tileB[16][16];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float sum = 0;
    int tile_cnt = (n + 15)/16;
    for(int i = 0; i < tile_cnt; i++){
        if (tid < 128){ // Load compressed matrix with first 128 threads
            int a_row = tid/8;
            int a_col = tid % 8;

            int global_a_row = blockIdx.y * blockDim.y + a_row;
            int global_a_col = 8 * i + a_col;
            
            if (global_a_row < n && global_a_col < n/2) tilecompressed[a_row][a_col] = compressed[index(global_a_row, global_a_col, n/2)];
            else tilecompressed[a_row][a_col] = 0;
        } else { // Load metadata with the rest of the threads
            int a_row = (tid - 128) / 8;
            int a_col = tid%8;
            
            int global_a_row = blockIdx.y * blockDim.y + a_row;
            int global_a_col = 8 * i + a_col;
            if (global_a_row < n && global_a_col < n/2) tilemetadata[a_row][a_col] = metadata[index(global_a_row, global_a_col, n/2)];
            else tilemetadata[a_row][a_col] = 0;
        }

        if (col < n && threadIdx.y + blockDim.y * i < n){
            tileB[threadIdx.y][threadIdx.x] = B[index(threadIdx.y + blockDim.y * i, col, n)];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();
        for(int k = 0; k < 4; k++){
            float val1 = tilecompressed[threadIdx.y][k * 2];
            float val2 = tilecompressed[threadIdx.y][k * 2 + 1];

            int offset1 = tilemetadata[threadIdx.y][k * 2];
            int offset2 = tilemetadata[threadIdx.y][k * 2 + 1];

            int k1 = k * 4 + offset1;
            int k2 = k * 4 + offset2;

            sum += val1 * tileB[k1][threadIdx.x];
            sum += val2 * tileB[k2][threadIdx.x];
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
    float* A_sparse = 0;
    float* B = 0;
    float* C = 0;
    cudaMallocHost(&A_sparse, sizeof(float) * n * n);
    cudaMallocHost(&B, sizeof(float) * n * n);
    cudaMallocHost(&C, sizeof(float) * n * n);

    Mat_rand_2_4(A_sparse, n);
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
    
    cudaMemcpy(A_gpu, A_sparse, sizeof(float) * n * n, cudaMemcpyDefault);
    cudaMemcpy(B_gpu, B, sizeof(float) * n * n, cudaMemcpyDefault);
    cudaMemset(C_gpu, 0, sizeof(float) * n * n);
    
    // Run Matrix Multiplication on GPU with Shared Memory between threads in Blocks
    auto start = std::chrono::steady_clock::now();   
    Mat_mult_shared<<<Blocks, threadsPerBlock>>>(A_gpu, B_gpu, C_gpu, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();   
    std::cout << "Matrix mult for GPU with shared Memory: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Copy result from Devise to Host
    cudaMemcpy(C, C_gpu, sizeof(float) * n * n, cudaMemcpyDefault);

    // Convert A_sparse to a compressed matrix and metadata
    float* compressed_gpu = 0;
    short* metadata_gpu = 0;
    cudaMalloc(&compressed_gpu, sizeof(float) * n * n / 2);
    cudaMalloc(&metadata_gpu, sizeof(short) * n * n / 2);

    cudaMemset(&compressed_gpu, 0, sizeof(float) * n * n / 2);

    int prune_thread = 256;
    int prune_block = (n * n / 4 + prune_thread - 1) / prune_thread;
    prune_2_4<<<prune_block, prune_thread>>>(A_gpu, compressed_gpu, metadata_gpu, n);
    cudaDeviceSynchronize();
    
    // Run SpMM on GPU
    start = std::chrono::steady_clock::now();   
    SpMM<<<Blocks, threadsPerBlock>>>(compressed_gpu, metadata_gpu, B_gpu, C_gpu, n);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();   
    std::cout << "SpMM for GPU with shared Memory: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    
    // Copy result from Device to Host and compare
    float* gpu_result = 0;
    cudaMallocHost(&gpu_result, sizeof(float) * n * n);
    cudaMemcpy(gpu_result, C_gpu, sizeof(float) * n * n, cudaMemcpyDefault);
    // mat_print(gpu_result, n);

    match(gpu_result, C, n);
}