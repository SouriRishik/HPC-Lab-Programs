#include <stdio.h>
#include <omp.h>

int main() {
    double num1, num2;
    double add, sub, mul, div;
    int div_valid = 1;

    printf("Enter two numbers: ");
    scanf("%lf %lf", &num1, &num2);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            add = num1 + num2;
            printf("Thread %d: Addition = %.2lf\n", omp_get_thread_num(), add);
        }

        #pragma omp section
        {
            sub = num1 - num2;
            printf("Thread %d: Subtraction = %.2lf\n", omp_get_thread_num(), sub);
        }

        #pragma omp section
        {
            mul = num1 * num2;
            printf("Thread %d: Multiplication = %.2lf\n", omp_get_thread_num(), mul);
        }

        #pragma omp section
        {
            if(num2 != 0)
                div = num1 / num2;
            else {
                div_valid = 0;
                div = 0;
            }
            if (div_valid)
                printf("Thread %d: Division = %.2lf\n", omp_get_thread_num(), div);
            else
                printf("Thread %d: Division by zero is not allowed.\n", omp_get_thread_num());
        }
    }

    printf("\nResults:\n");
    printf("Addition: %.2lf\n", add);
    printf("Subtraction: %.2lf\n", sub);
    printf("Multiplication: %.2lf\n", mul);
    if(div_valid)
        printf("Division: %.2lf\n", div);
    else
        printf("Division: Error (division by zero)\n");

    return 0;
}