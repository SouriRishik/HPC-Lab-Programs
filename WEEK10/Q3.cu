#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void csrSpMV(int num_rows, const int *d_row_ptr, const int *d_col_ind,
                        const float *d_val, const float *d_x, float *d_y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float sum = 0.0f;
        int start_idx = d_row_ptr[row];
        int end_idx = d_row_ptr[row + 1];

        for (int i = start_idx; i < end_idx; ++i) {
            sum += d_val[i] * d_x[d_col_ind[i]];
        }
        d_y[row] = sum;
    }
}

int main() {
    int num_rows = 5;
    int num_cols = 5;
    int num_non_zeros = 9;

    int h_row_ptr[] = {0, 2, 4, 5, 7, 9};
    int h_col_ind[] = {0, 2, 1, 3, 4, 0, 4, 1, 3};
    float h_val[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float *h_y = (float *)malloc(num_rows * sizeof(float));

    int *d_row_ptr, *d_col_ind;
    float *d_val, *d_x, *d_y;

    cudaMalloc((void **)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void **)&d_col_ind, num_non_zeros * sizeof(int));
    cudaMalloc((void **)&d_val, num_non_zeros * sizeof(float));
    cudaMalloc((void **)&d_x, num_cols * sizeof(float));
    cudaMalloc((void **)&d_y, num_rows * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind, num_non_zeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_val, num_non_zeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;

    csrSpMV<<<blocks_per_grid, threads_per_block>>>(num_rows, d_row_ptr, d_col_ind, d_val, d_x, d_y);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result vector y:\n");
    for (int i = 0; i < num_rows; ++i) {
        printf("%.2f ", h_y[i]);
    }
    printf("\n");

    free(h_y);
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}