#include <stdio.h>
#include <omp.h>
#include <ctype.h>
#include <string.h>

int main() {
    char str[100];
    printf("Enter a string: ");
    fgets(str, sizeof(str), stdin);
    size_t len = strlen(str);
    if (len > 0 && str[len - 1] == '\n') {
        str[len - 1] = '\0';
        len--;
    }

    #pragma omp parallel num_threads(len)
    {
        int tid = omp_get_thread_num();
        char ch = str[tid];
        if (isupper(ch))
            str[tid] = tolower(ch);
        else if (islower(ch))
            str[tid] = toupper(ch);

        printf("Thread_Id: %d, Toggled Char: %c\n", tid, str[tid]);
    }

    printf("Resulting string: %s\n", str);
    return 0;
}