#include <stdio.h>
#include <string.h>
#include <omp.h>

int main() {
    char str[100];
    int len, i;
    printf("Please Enter any String: ");
    fgets(str, sizeof(str), stdin);
    len = strlen(str);
    if (str[len - 1] == '\n') {
        str[len - 1] = '\0';
        len--;
    }
    #pragma omp parallel for
    for (i = 0; i < len; i++) {
        int tid = omp_get_thread_num();
        if (str[i] >= 'A' && str[i] <= 'Z')
            str[i] = str[i] + 32;
        else if (str[i] >= 'a' && str[i] <= 'z')
            str[i] = str[i] - 32;
        printf("Thread %d processed character %d\n", tid, i);
    }
    printf("\nToggled string: %s\n", str);
    return 0;
}
