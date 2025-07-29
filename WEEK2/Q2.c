#include <stdio.h>
#include <omp.h>
#include <math.h>

int main() {
    int n;
    printf("Enter the number of threads/integers: ");
    scanf("%d", &n);
    omp_set_num_threads(n);
    printf("Calculating pow(i, x) where i is integer (1 to n), x is thread id (0 to n-1):\n");
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int i = thread_id + 1;
        double result = pow(i, thread_id);
        printf("Thread %d: pow(%d, %d) = %.2f\n", thread_id, i, thread_id, result);
    }
    return 0;
}