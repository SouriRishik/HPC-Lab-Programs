#include <stdio.h>
#include <cuda_runtime.h>

__global__ void mulRow(int *A, int *B, int *C, int M, int N) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < M) {
        for (int col = 0; col < N; col++)
            C[row * N + col] = A[row * N + col] * B[row * N + col];
    }
}

__global__ void mulCol(int *A, int *B, int *C, int M, int N) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < N) {
        for (int row = 0; row < M; row++)
            C[row * N + col] = A[row * N + col] * B[row * N + col];
    }
}

__global__ void mulElement(int *A, int *B, int *C, int total) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < total) {
        C[idx] = A[idx] * B[idx];
    }
}

void printMatrix(int *mat, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", mat[i * N + j]);
        printf("\n");
    }
    printf("\n");
}

int main() {
    int M = 3, N = 3;
    int A[] = {1,2,3,4,5,6,7,8,9};
    int B[] = {9,8,7,6,5,4,3,2,1};
    int C[M*N];

    int *dA, *dB, *dC;
    cudaMalloc(&dA, M*N*sizeof(int));
    cudaMalloc(&dB, M*N*sizeof(int));
    cudaMalloc(&dC, M*N*sizeof(int));

    cudaMemcpy(dA, A, M*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, M*N*sizeof(int), cudaMemcpyHostToDevice);
    
    int n;
    printf("Enter mode (0: row-wise, 1: column-wise, 2: element-wise): ");
    scanf("%d", &n);
    int mode = n;

    if (mode == 0) {
        int threads = 32;
        int blocks = (M + threads - 1) / threads;
        mulRow<<<blocks, threads>>>(dA, dB, dC, M, N);
    } else if (mode == 1) {
        int threads = 32;
        int blocks = (N + threads - 1) / threads;
        mulCol<<<blocks, threads>>>(dA, dB, dC, M, N);
    } else {
        int total = M * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        mulElement<<<blocks, threads>>>(dA, dB, dC, total);
    }

    cudaMemcpy(C, dC, M*N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result matrix (mode = %d):\n", mode);
    printMatrix(C, M, N);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}