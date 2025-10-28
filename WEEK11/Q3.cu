#include <stdio.h>
#include <cuda.h>

__global__ void inclusiveScan(int *d_out, int *d_in, int n)
{
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int offset = 1;
    int ai = thid;
    int bi = thid + (n / 2);
    temp[ai] = d_in[ai];
    temp[bi] = d_in[bi];
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();
    temp[n - 1] = temp[n - 1];
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    d_out[ai] = temp[ai];
    d_out[bi] = temp[bi];
}

int main()
{
    int n = 8;
    int h_in[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_out[8];
    int *d_in, *d_out;
    cudaMalloc((void **)&d_in, n * sizeof(int));
    cudaMalloc((void **)&d_out, n * sizeof(int));
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);
    inclusiveScan<<<1, n / 2, n * sizeof(int)>>>(d_out, d_in, n);
    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        printf("%d ", h_out[i]);
    printf("\n");
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}