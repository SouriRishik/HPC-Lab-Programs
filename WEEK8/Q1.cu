#include <stdio.h>
#include <string.h>

__global__ void repeatString(char *S, char *out, int len, int N)
{
    int idx = threadIdx.x;
    if (idx < N)
    {
        for (int i = 0; i < len; i++)
        {
            out[idx * len + i] = S[i];
        }
    }
}

int main()
{
    char S[100];
    int N;
    printf("Enter string: ");
    scanf("%s", S);
    printf("Enter N: ");
    scanf("%d", &N);

    int len = strlen(S);
    int outLen = len * N;
    char *d_S, *d_out;
    char *out = (char *)malloc(outLen + 1);

    cudaMalloc(&d_S, len);
    cudaMalloc(&d_out, outLen);

    cudaMemcpy(d_S, S, len, cudaMemcpyHostToDevice);

    repeatString<<<1, N>>>(d_S, d_out, len, N);

    cudaMemcpy(out, d_out, outLen, cudaMemcpyDeviceToHost);

    out[outLen] = '\0';
    printf("Output String: %s\n", out);

    cudaFree(d_S);
    cudaFree(d_out);
    free(out);
    return 0;
}