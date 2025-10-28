#include <stdio.h>
#include <cuda.h>

#define MAX_FILTER_SIZE 64

__constant__ float d_filter[MAX_FILTER_SIZE];

__global__ void convolution1D(float *d_signal, float *d_output, int signal_size, int filter_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half_filter = filter_size / 2;

    if (i < signal_size)
    {
        float sum = 0.0f;
        for (int j = 0; j < filter_size; j++)
        {
            int signal_index = i - half_filter + j;
            if (signal_index >= 0 && signal_index < signal_size)
            {
                sum += d_signal[signal_index] * d_filter[j];
            }
        }
        d_output[i] = sum;
    }
}

int main()
{
    int signal_size, filter_size;

    printf("Enter signal size: ");
    scanf("%d", &signal_size);

    printf("Enter filter size (<= %d): ", MAX_FILTER_SIZE);
    scanf("%d", &filter_size);

    if (filter_size > MAX_FILTER_SIZE)
    {
        printf("Filter size too large! Max allowed is %d\n", MAX_FILTER_SIZE);
        return -1;
    }

    float *h_signal = (float *)malloc(signal_size * sizeof(float));
    float *h_filter = (float *)malloc(filter_size * sizeof(float));
    float *h_output = (float *)malloc(signal_size * sizeof(float));

    printf("Enter signal values (%d numbers):\n", signal_size);
    for (int i = 0; i < signal_size; i++)
        scanf("%f", &h_signal[i]);

    printf("Enter filter values (%d numbers):\n", filter_size);
    for (int i = 0; i < filter_size; i++)
        scanf("%f", &h_filter[i]);

    float *d_signal, *d_output;
    cudaMalloc((void **)&d_signal, signal_size * sizeof(float));
    cudaMalloc((void **)&d_output, signal_size * sizeof(float));

    cudaMemcpy(d_signal, h_signal, signal_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter, h_filter, filter_size * sizeof(float));

    int blockSize = 256;
    int gridSize = (signal_size + blockSize - 1) / blockSize;

    convolution1D<<<gridSize, blockSize>>>(d_signal, d_output, signal_size, filter_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, signal_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nConvolution Output:\n");
    for (int i = 0; i < signal_size; i++)
        printf("%.2f ", h_output[i]);
    printf("\n");

    cudaFree(d_signal);
    cudaFree(d_output);
    free(h_signal);
    free(h_filter);
    free(h_output);

    return 0;
}
