#include <stdio.h>
#include <cuda.h>

#define MAX_MASK_SIZE 64
#define TILE_SIZE 256

__constant__ float d_mask[MAX_MASK_SIZE];

__global__ void convolution1D_tiled(float *d_input, float *d_output, int N, int M)
{
    __shared__ float tile[TILE_SIZE + MAX_MASK_SIZE - 1];
    int tx = threadIdx.x;
    int start = blockIdx.x * TILE_SIZE;
    int i = start + tx;
    int halo = M / 2;
    int input_index = i - halo;

    if (input_index >= 0 && input_index < N)
        tile[tx] = d_input[input_index];
    else
        tile[tx] = 0.0f;

    if (tx < M - 1)
    {
        int halo_index = start + TILE_SIZE + tx - halo;
        if (halo_index >= 0 && halo_index < N)
            tile[TILE_SIZE + tx] = d_input[halo_index];
        else
            tile[TILE_SIZE + tx] = 0.0f;
    }

    __syncthreads();

    if (i < N)
    {
        float sum = 0.0f;
        for (int j = 0; j < M; j++)
            sum += tile[tx + j] * d_mask[j];
        d_output[i] = sum;
    }
}

int main()
{
    int N, M;
    printf("Enter input size: ");
    scanf("%d", &N);
    printf("Enter mask size (<= %d): ", MAX_MASK_SIZE);
    scanf("%d", &M);

    if (M > MAX_MASK_SIZE)
        return -1;

    float *h_input = (float *)malloc(N * sizeof(float));
    float *h_mask = (float *)malloc(M * sizeof(float));
    float *h_output = (float *)malloc(N * sizeof(float));

    printf("Enter input values:\n");
    for (int i = 0; i < N; i++)
        scanf("%f", &h_input[i]);
    printf("Enter mask values:\n");
    for (int i = 0; i < M; i++)
        scanf("%f", &h_mask[i]);

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask, M * sizeof(float));

    int gridSize = (N + TILE_SIZE - 1) / TILE_SIZE;
    convolution1D_tiled<<<gridSize, TILE_SIZE>>>(d_input, d_output, N, M);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output:\n");
    for (int i = 0; i < N; i++)
        printf("%.2f ", h_output[i]);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_mask);
    free(h_output);
    return 0;
}
