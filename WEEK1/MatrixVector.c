#include <stdio.h>
#include <omp.h>

int main() {
    int A[5][5], X[5], Y[5] = {0};

    printf("Enter 5x5 matrix A:\n");
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            scanf("%d", &A[i][j]);

    printf("Enter 5-element vector X:\n");
    for (int i = 0; i < 5; i++)
        scanf("%d", &X[i]);

    #pragma omp parallel for
    for (int i = 0; i < 5; i++) {
        Y[i] = 0;
        for (int j = 0; j < 5; j++)
            Y[i] += A[i][j] * X[j];
    }

    printf("Resultant vector Y = A × X:\n");
    for (int i = 0; i < 5; i++)
        printf("%d ", Y[i]);
    printf("\n");

    return 0;
}
