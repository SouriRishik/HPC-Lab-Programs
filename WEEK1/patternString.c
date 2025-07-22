#include <stdio.h>
#include <string.h>
#include <omp.h>

int main() {
    char input_word[100];
    printf("Enter a word: ");
    scanf("%s", input_word);
    int n = strlen(input_word);
    printf("Output: ");
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int count = i + 1;
        for (int j = 0; j < count; j++) {
            printf("%c", input_word[i]);
        }
    }
    printf("\n");
    return 0;
}

