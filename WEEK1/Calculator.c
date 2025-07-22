#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main() {
    float a, b, sum, diff, product, quotient;
    int thread_id;

    printf("Enter two numbers: ");
    scanf("%f %f", &a, &b);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            thread_id = omp_get_thread_num();
            sum = a + b;
            printf("Thread %d: Sum = %.2f\n", thread_id, sum);
        }
        #pragma omp section
        {
            thread_id = omp_get_thread_num();
            diff = a - b;
            printf("Thread %d: Difference = %.2f\n", thread_id, diff);
        }
        #pragma omp section
        {
            thread_id = omp_get_thread_num();
            product = a * b;
            printf("Thread %d: Product = %.2f\n", thread_id, product);
        }
        #pragma omp section
        {
            thread_id = omp_get_thread_num();
            if (b != 0) {
                quotient = a / b;
                printf("Thread %d: Quotient = %.2f\n", thread_id, quotient);
            } else {
                printf("Thread %d: Cannot divide by zero.\n", thread_id);
            }
        }
    }

    return 0;
}
