#include <stdio.h>
#include <string.h>
#include <omp.h>

int main() {
    char s1[100];
    char s2[100];
    char resultant_string[200];

    printf("Enter the first string (S1): ");
    scanf("%s", s1);

    printf("Enter the second string (S2): ");
    scanf("%s", s2);

    int len1 = strlen(s1);
    int len2 = strlen(s2);

    if (len1 != len2) {
        printf("Error: Strings must be of the same length.\n");
        return 1;
    }
    #pragma omp parallel for
    for (int i = 0; i < len1; i++) {
        resultant_string[2 * i] = s1[i];
        resultant_string[2 * i + 1] = s2[i];
    }

    resultant_string[2 * len1] = '\0';

    printf("Resultant String: %s\n", resultant_string);

    return 0;
}
