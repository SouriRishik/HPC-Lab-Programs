#include <stdio.h>
#include <string.h>

__global__ void reverseAll(char *s, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = len - 1 - i;
    if (i < j) {
        char temp = s[i];
        s[i] = s[j];
        s[j] = temp;
    }
}

int main() {
    char str[512];
    printf("Enter string: ");
    fgets(str, sizeof(str), stdin);
    str[strcspn(str, "\n")] = '\0';

    int len = strlen(str);
    if (len == 0) { printf("Reversed: \n"); return 0; }

    char *d_str;
    cudaMalloc(&d_str, len);                
    cudaMemcpy(d_str, str, len, cudaMemcpyHostToDevice);

    int half = (len + 1) / 2;                
    int threads = 128;
    int blocks = (half + threads - 1) / threads;
    reverseAll<<<blocks, threads>>>(d_str, len);
    cudaDeviceSynchronize();                 

    cudaMemcpy(str, d_str, len, cudaMemcpyDeviceToHost);
    cudaFree(d_str);

    printf("Reversed: %s\n", str);
    return 0;
}