#include <stdio.h>
#include <omp.h>

#define SIZE 5

int main() {
    int A[SIZE][SIZE];
    int B[SIZE][SIZE];

    printf("Enter 25 elements for 5x5 matrix A:\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            scanf("%d", &A[i][j]);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        int max_val = A[i][0];
        int min_val = A[i][0];

        for (int j = 1; j < SIZE; j++) {
            if (A[i][j] > max_val)
                max_val = A[i][j];
            if (A[i][j] < min_val)
                min_val = A[i][j];
        }

        for (int j = 0; j < SIZE; j++) {
            if (i == j)
                B[i][j] = 0;          
            else if (j < i)
                B[i][j] = max_val;    
            else
                B[i][j] = min_val;    
        }
    }

    printf("Resultant matrix B:\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%d\t", B[i][j]);
        }
        printf("\n");
    }

    return 0;
}

