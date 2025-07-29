#include <stdio.h>
#include <omp.h>

int main() {
    int M = 4, N = 2;
    
    char A[M][N];
    int B[M][N];

    printf("Enter character matrix A (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf(" %c", &A[i][j]);
        }
    }

    printf("Enter integer matrix B (%dx%d):\n", M, N);
    int total_len = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%d", &B[i][j]);
            total_len += B[i][j];
        }
    }

    char STR[total_len + 1];
    STR[total_len] = '\0';

    int prefix[M * N + 1];
    prefix[0] = 0;
    int idx = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            prefix[idx + 1] = prefix[idx] + B[i][j];
            idx++;
        }
    }

    #pragma omp parallel for
    for (int k = 0; k < M * N; k++) {
        int i = k / N;
        int j = k % N;

        int start = prefix[k];
        int count = B[i][j];
        char ch = A[i][j];
        for (int l = 0; l < count; l++) {
            STR[start + l] = ch;
        }
    }

    printf("Output string STR: %s\n", STR);

    return 0;
}

