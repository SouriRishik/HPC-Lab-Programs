#include <stdio.h>
#include <stdlib.h>

__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0;

    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            Cvalue += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = Cvalue;
    }
}

int main() {
    int N = 1024;
    int size = N * N * sizeof(float);

    float *h_A, *h_B, *h_C;

    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = i;
        h_B[i] = i;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int blockSize = 16;
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result C[0][0]: %f\n", h_C[0]);
    printf("Result C[N-1][N-1]: %f\n", h_C[N*N - 1]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}