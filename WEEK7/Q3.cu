#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 16

__global__ void computeSine(const float *angles, float *sines, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sines[i] = sinf(angles[i]);
    }
}

void printArray(const char *label, float *arr, int n) {
    printf("%s: ", label);
    for (int i = 0; i < n; i++) printf("%.4f ", arr[i]);
    printf("\n");
}

int main() {
    float h_angles[N], h_sines[N];
    float *d_angles, *d_sines;
    size_t size = N * sizeof(float);

    for (int i = 0; i < N; i++) h_angles[i] = i * (3.14159265f / 8);

    cudaMalloc((void**)&d_angles, size);
    cudaMalloc((void**)&d_sines, size);

    cudaMemcpy(d_angles, h_angles, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    computeSine<<<gridSize, blockSize>>>(d_angles, d_sines, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_sines, d_sines, size, cudaMemcpyDeviceToHost);

    printArray("Angles (rad)", h_angles, N);
    printArray("Sine", h_sines, N);

    cudaFree(d_angles);
    cudaFree(d_sines);
    return 0;
}
