#include <stdio.h>
#include <omp.h>

int fibonacci(int n) {
    if (n <= 1)
        return n;
    int a = 0, b = 1, c;
    for (int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

int main() {
    int A[] = {10, 13, 5, 6};
    int n = sizeof(A) / sizeof(A[0]);
    int fib[n];

    #pragma omp parallel num_threads(n)
    {
        int tid = omp_get_thread_num();
        fib[tid] = fibonacci(A[tid]);
        printf("Thread_Id: %d, Number: %d, Fibonacci: %d\n", tid, A[tid], fib[tid]);
    }

    printf("Fibonacci results: ");
    for (int i = 0; i < n; i++)
        printf("%d ", fib[i]);
    printf("\n");

    return 0;
}