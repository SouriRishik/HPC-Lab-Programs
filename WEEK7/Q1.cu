#include <stdio.h>
#include <cuda_runtime.h>

#define N 4

__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
        if (i < 5) {
            printf("[Device] i=%d blockIdx.x=%d threadIdx.x=%d blockDim.x=%d gridDim.x=%d\n",
                i, blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
        }
    }
}

void checkResult(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        if (C[i] != A[i] + B[i]) {
            printf("Mismatch at %d: %f != %f\n", i, C[i], A[i] + B[i]);
            return;
        }
    }
    printf("Result correct!\n");
}

void printArrays(float *A, float *B, float *C, int n, int num) {
    printf("Index\tA\t\tB\t\tC\n");
    int limit = n < num ? n : num;
    for (int i = 0; i < limit; i++) {
        printf("%d\t%.2f\t\t%.2f\t\t%.2f\n", i, A[i], B[i], C[i]);
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = (N - i) * 1.0f;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    printf("Case a) Block size = %d, 1 block, %d threads\n", N, N);
    printf("[Host] Launch config: gridDim.x=1, blockDim.x=%d, total threads=%d\n", N, N);
    vectorAdd<<<1, N>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    checkResult(h_A, h_B, h_C, N);
    printArrays(h_A, h_B, h_C, N, 10);

    printf("Case b) %d threads, 1 block\n", N);
    printf("[Host] Launch config: gridDim.x=1, blockDim.x=%d, total threads=%d\n", N, N);
    vectorAdd<<<1, N>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    checkResult(h_A, h_B, h_C, N);
    printArrays(h_A, h_B, h_C, N, 10);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    printf("Case c) Block size = 256, grid size = %d\n", gridSize);
    printf("[Host] Launch config: gridDim.x=%d, blockDim.x=%d, total threads=%d\n", gridSize, blockSize, gridSize*blockSize);
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    checkResult(h_A, h_B, h_C, N);
    printArrays(h_A, h_B, h_C, N, 10);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
