#include <stdio.h>
#include <omp.h>

int main() {
    printf("OpenMP Fork-Join and SPMD Patterns Demo\n");

    // a. Fork-Join pattern using a single parallel directive
    printf("\n(a) Fork-Join with one parallel region:\n");
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("  Hello from thread %d\n", tid);
    }
    printf("  Back to single thread after join.\n");

    // b. Fork-Join pattern using multiple parallel directives and changing thread count
    printf("\n(b) Fork-Join with multiple parallel regions and thread count changes:\n");
    omp_set_num_threads(2);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("  [2 threads] Thread %d\n", tid);
    }
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        printf("  [4 threads] Thread %d\n", tid);
    }

    // c. SPMD pattern using two basic OpenMP commands
    printf("\n(c) SPMD Pattern with parallel region and thread ID:\n");
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("  SPMD: This is thread %d of %d\n", tid, nthreads);
    }

    return 0;
}
