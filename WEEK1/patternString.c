#include <stdio.h>
#include <string.h>
#include <omp.h>

int main() {
    char input_word[100];
    printf("Enter a word: ");
    scanf("%s", input_word);

    int n = strlen(input_word);
    char output[1000] = "";

    #pragma omp parallel
    {
        char local[100] = "";
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            int count = i + 1;
            for (int j = 0; j < count; j++) {
                local[j] = input_word[i];
            }
            local[count] = '\0';
            #pragma omp critical
            strcat(output, local);
        }
    }

    printf("Output: %s\n", output);
    return 0;
}
