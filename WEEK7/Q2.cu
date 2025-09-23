#include <stdio.h>
#include <cuda_runtime.h>

#define N 16
#define M 3    

__global__ void convolution1D(float *input, float *mask, float *output, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = m / 2;
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < m; j++) {
            int idx = i - radius + j;
            if (idx >= 0 && idx < n) {
                sum += input[idx] * mask[j];
            }
        }
        output[i] = sum;
    }
}

void printArray(const char *label, float *arr, int n) {
    printf("%s: ", label);
    for (int i = 0; i < n; i++) printf("%.4f ", arr[i]);
    printf("\n");
}

int main() {
    float h_input[N], h_mask[M], h_output[N];
    float *d_input, *d_mask, *d_output;
    size_t sizeInput = N * sizeof(float);
    size_t sizeMask = M * sizeof(float);

    for (int i = 0; i < N; i++) h_input[i] = i + 1;
    h_mask[0] = 0.33f;
    h_mask[1] = 0.33f;
    h_mask[2] = 0.33f;

    cudaMalloc((void**)&d_input, sizeInput);
    cudaMalloc((void**)&d_mask, sizeMask);
    cudaMalloc((void**)&d_output, sizeInput);

    cudaMemcpy(d_input, h_input, sizeInput, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, sizeMask, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    convolution1D<<<gridSize, blockSize>>>(d_input, d_mask, d_output, N, M);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeInput, cudaMemcpyDeviceToHost);

    printArray("Input", h_input, N);
    printArray("Mask", h_mask, M);
    printArray("Output", h_output, N);

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
    return 0;
}
