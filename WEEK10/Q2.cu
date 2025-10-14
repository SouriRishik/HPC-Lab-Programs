#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void convolution2DKernel(float* d_input, float* d_mask, float* d_output, int inputRows, int inputCols, int maskRows, int maskCols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < inputRows && col < inputCols) {
        float sum = 0.0f;
        int maskHalfRows = maskRows / 2;
        int maskHalfCols = maskCols / 2;

        for (int i = 0; i < maskRows; ++i) {
            for (int j = 0; j < maskCols; ++j) {
                int inputRow = row - maskHalfRows + i;
                int inputCol = col - maskHalfCols + j;

                if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols) {
                    sum += d_input[inputRow * inputCols + inputCol] * d_mask[i * maskCols + j];
                }
            }
        }
        d_output[row * inputCols + col] = sum;
    }
}

void convolution2D(float* h_input, float* h_mask, float* h_output, int inputRows, int inputCols, int maskRows, int maskCols) {
    float* d_input;
    float* d_mask;
    float* d_output;

    size_t inputSize = inputRows * inputCols * sizeof(float);
    size_t maskSize = maskRows * maskCols * sizeof(float);

    cudaMalloc((void**)&d_input, inputSize);
    cudaMalloc((void**)&d_mask, maskSize);
    cudaMalloc((void**)&d_output, inputSize);

    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, maskSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((inputCols + blockSize.x - 1) / blockSize.x, (inputRows + blockSize.y - 1) / blockSize.y);

    convolution2DKernel<<<gridSize, blockSize>>>(d_input, d_mask, d_output, inputRows, inputCols, maskRows, maskCols);

    cudaMemcpy(h_output, d_output, inputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
}

int main() {
    int inputRows = 5;
    int inputCols = 5;
    int maskRows = 3;
    int maskCols = 3;

    float* h_input = (float*)malloc(inputRows * inputCols * sizeof(float));
    float* h_mask = (float*)malloc(maskRows * maskCols * sizeof(float));
    float* h_output = (float*)malloc(inputRows * inputCols * sizeof(float));

    for (int i = 0; i < inputRows * inputCols; ++i) {
        h_input[i] = i + 1;
    }

    for (int i = 0; i < maskRows * maskCols; ++i) {
        h_mask[i] = 1.0f;
    }
    
    convolution2D(h_input, h_mask, h_output, inputRows, inputCols, maskRows, maskCols);

    printf("Input Array:\n");
    for (int i = 0; i < inputRows; ++i) {
        for (int j = 0; j < inputCols; ++j) {
            printf("%.1f ", h_input[i * inputCols + j]);
        }
        printf("\n");
    }

    printf("\nMask Array:\n");
    for (int i = 0; i < maskRows; ++i) {
        for (int j = 0; j < maskCols; ++j) {
            printf("%.1f ", h_mask[i * maskCols + j]);
        }
        printf("\n");
    }

    printf("\nOutput Array:\n");
    for (int i = 0; i < inputRows; ++i) {
        for (int j = 0; j < inputCols; ++j) {
            printf("%.1f ", h_output[i * inputCols + j]);
        }
        printf("\n");
    }

    free(h_input);
    free(h_mask);
    free(h_output);

    return 0;
}