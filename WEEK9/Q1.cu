#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void transformMatrix(int *mat, int M, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M && c < N) {
        mat[r * N + c] = (int)powf(mat[r * N + c], r + 1);
    }
}

int main() {
    int M, N;
    scanf("%d%d", &M, &N);
    int *h_mat = (int*)malloc(M * N * sizeof(int));
    for (int i = 0; i < M * N; i++) scanf("%d", &h_mat[i]);

    int *d_mat;
    cudaMalloc(&d_mat, M * N * sizeof(int));
    cudaMemcpy(d_mat, h_mat, M * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16,16), grid((N + 15)/16, (M + 15)/16);
    transformMatrix<<<grid, block>>>(d_mat, M, N);
    cudaMemcpy(h_mat, d_mat, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) printf("%d ", h_mat[i * N + j]);
        printf("\n");
    }

    free(h_mat);
    cudaFree(d_mat);
    return 0;
}