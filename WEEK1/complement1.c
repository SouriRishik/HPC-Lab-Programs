#include <stdio.h>
#include <omp.h>
#include <math.h>

int decToBinaryDigits(int n) {
    int bin = 0, place = 1;
    while (n > 0) {
        bin += (n % 2) * place;
        n /= 2;
        place *= 10;
    }
    return bin;
}

int countDigits(int n) {
    int count = 0;
    if (n == 0) return 1;
    while (n > 0) {
        n /= 10;
        count++;
    }
    return count;
}

int onesComplementBinaryDigits(int binary) {
    int result = 0, place = 1;
    while (binary > 0) {
        int bit = binary % 10;
        int comp_bit = (bit == 0) ? 1 : 0;
        result += comp_bit * place;
        binary /= 10;
        place *= 10;
    }
    return result;
}

int binaryDigitsToDecimal(int binary) {
    int decimal = 0, base = 1;
    while (binary > 0) {
        int last_digit = binary % 10;
        decimal += last_digit * base;
        base *= 2;
        binary /= 10;
    }
    return decimal;
}

int main() {
    int M, N;
    printf("Enter number of rows (M): ");
    scanf("%d", &M);
    printf("Enter number of columns (N): ");
    scanf("%d", &N);

    int A[M][N], B[M][N], D[M][N];

    printf("Enter matrix A (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%d", &A[i][j]);
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0 || i == M-1 || j == 0 || j == N-1) {
                B[i][j] = A[i][j];
                D[i][j] = A[i][j];
            } else {
                int binary = decToBinaryDigits(A[i][j]);         
                int onesComp = onesComplementBinaryDigits(binary); 
                B[i][j] = onesComp;                                
                D[i][j] = binaryDigitsToDecimal(onesComp);        
            }
        }
    }

    printf("Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", B[i][j]);
        }
        printf("\n");
    }

    printf("Matrix D:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", D[i][j]);
        }
        printf("\n");
    }

    return 0;
}

