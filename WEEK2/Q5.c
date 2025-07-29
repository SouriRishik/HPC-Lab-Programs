#include <stdio.h>
#include <omp.h>

int is_prime(int num) {
    if (num < 2) return 0;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0)
            return 0;
    }
    return 1;
}

int main() {
    int start, end;
    printf("Enter the starting number: ");
    scanf("%d", &start);
    printf("Enter the ending number: ");
    scanf("%d", &end);

    printf("Prime numbers between %d and %d are:\n", start, end);

    #pragma omp parallel for schedule(dynamic)
    for (int i = start; i <= end; i++) {
        if (is_prime(i)) {
            #pragma omp critical
            printf("%d ", i);
        }
    }
    printf("\n");
    return 0;
}