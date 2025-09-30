#include <stdio.h>
#include <string.h>

__global__ void countWord(const char *sentence, int sentLen,
                          const char *word, int wordLen, int *count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + wordLen > sentLen)
        return;

    if (i > 0 && sentence[i - 1] != ' ')
        return;
    if (sentence[i] != word[0])
        return;

    for (int j = 0; j < wordLen; ++j)
    {
        if (sentence[i + j] != word[j])
            return;
    }
    if (i + wordLen < sentLen && sentence[i + wordLen] != ' ')
        return;
    
    atomicAdd(count, 1);
}

int main()
{
    char sentence[256];
    char word[64];
    printf("Enter sentence: ");
    fgets(sentence, sizeof(sentence), stdin);
    sentence[strcspn(sentence, "\n")] = '\0';
    printf("Enter word to count: ");
    scanf("%63s", word);

    int sentLen = strlen(sentence);
    int wordLen = strlen(word);

    char *d_sentence, *d_word;
    int *d_count;
    int h_count = 0;
    cudaMalloc(&d_sentence, sentLen);
    cudaMalloc(&d_word, wordLen);
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_sentence, sentence, sentLen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, wordLen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (sentLen + threads - 1) / threads;
    countWord<<<blocks, threads>>>(d_sentence, sentLen, d_word, wordLen, d_count);

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Occurrences: %d\n", h_count);

    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);
    return 0;
}