#include <stdio.h>
#include <string.h>

__global__ void reverseWords(char *str, int *word_starts, int *word_ends, int N)
{
    int idx = threadIdx.x;
    if (idx < N)
    {
        int start = word_starts[idx];
        int end = word_ends[idx];
        while (start < end)
        {
            char temp = str[start];
            str[start++] = str[end--];
            str[end] = temp;
        }
    }
}

int main()
{
    char str[256];
    printf("Enter a string: ");
    scanf(" %[^\n]", str);

    int word_starts[50], word_ends[50], N = 0;
    int len = strlen(str);
    int i = 0;
    while (i < len)
    {
        while (i < len && str[i] == ' ')
            i++;
        if (i < len)
            word_starts[N] = i;
        while (i < len && str[i] != ' ')
            i++;
        if (i > 0 && str[i - 1] != ' ')
            word_ends[N] = i - 1;
        if (i <= len)
            N++;
    }

    char *d_str;
    int *d_starts, *d_ends;
    cudaMalloc(&d_str, len);
    cudaMalloc(&d_starts, N * sizeof(int));
    cudaMalloc(&d_ends, N * sizeof(int));

    cudaMemcpy(d_str, str, len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_starts, word_starts, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ends, word_ends, N * sizeof(int), cudaMemcpyHostToDevice);

    reverseWords<<<1, N>>>(d_str, d_starts, d_ends, N);

    cudaMemcpy(str, d_str, len, cudaMemcpyDeviceToHost);

    printf("Output: %s\n", str);

    cudaFree(d_str);
    cudaFree(d_starts);
    cudaFree(d_ends);
    return 0;
}