#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 8;
    size_t size = N * N * sizeof(float);
    float *hA = (float*)malloc(size);
    float *hB = (float*)malloc(size);
    float *hC = (float*)malloc(size);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) hA[i * N + j] = (float)(i + j);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) hB[i * N + j] = (i == j) ? 1.0f : 0.0f;
    float *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matMul<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();
    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) printf("%6.1f ", hC[i * N + j]);
        printf("\n");
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);
    return 0;
}
