#include <stdio.h>
#include <cuda_runtime.h>

__global__ void processMatrix(int *A, int *B, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        int idx = i * N + j;

        if (i == 0 || i == M-1 || j == 0 || j == N-1) {
            B[idx] = A[idx];
        } else {

            int value = A[idx];
            int binary = 0, place = 1;

            while (value > 0) {
                binary += (value % 2) * place;
                value /= 2;
                place *= 10;
            }

            int result = 0;
            place = 1;
            while (binary > 0) {
                int bit = binary % 10;
                int comp_bit = (bit == 0) ? 1 : 0;
                result += comp_bit * place;
                binary /= 10;
                place *= 10;
            }
            B[idx] = result;
        }
    }
}

int main() {
    int M, N;
    printf("Enter number of rows (M): ");
    scanf("%d", &M);
    printf("Enter number of columns (N): ");
    scanf("%d", &N);

    int *A = (int*)malloc(M * N * sizeof(int));
    int *B = (int*)malloc(M * N * sizeof(int));

    printf("Enter matrix A (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%d", &A[i * N + j]);
        }
    }

    int *d_A, *d_B;
    cudaMalloc(&d_A, M * N * sizeof(int));
    cudaMalloc(&d_B, M * N * sizeof(int));

    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    processMatrix<<<gridSize, blockSize>>>(d_A, d_B, M, N);

    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", B[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(A);
    free(B);

    return 0;
}
