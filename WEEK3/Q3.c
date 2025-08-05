#include <stdio.h>
#include <omp.h>

#define MAX_SIZE 100000
#define MAX_THREADS 8

void vector_add(int *a, int *b, int *res, int n) {
    for (int i = 0; i < n; i++)
        res[i] = a[i] + b[i];
}

void vector_sub(int *a, int *b, int *res, int n) {
    for (int i = 0; i < n; i++)
        res[i] = a[i] - b[i];
}

void vector_mul(int *a, int *b, int *res, int n) {
    for (int i = 0; i < n; i++)
        res[i] = a[i] * b[i];
}

int main() {
    int n;
    int a[MAX_SIZE], b[MAX_SIZE];
    int add_res[MAX_SIZE], sub_res[MAX_SIZE], mul_res[MAX_SIZE];
    int thread_counts[] = {1, 2, 4, 8};
    int num_threads = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("Enter vector size (max %d): ", MAX_SIZE);
    scanf("%d", &n);
    if (n <= 0 || n > MAX_SIZE) {
        printf("Invalid size.\n");
        return 1;
    }

    printf("Enter vector A elements:\n");
    for (int i = 0; i < n; i++) scanf("%d", &a[i]);

    printf("Enter vector B elements:\n");
    for (int i = 0; i < n; i++) scanf("%d", &b[i]);

    printf("\nVector Operations using OpenMP Task Parallelism\n");
    printf("%10s %15s %15s %15s\n", "Threads", "Time (s)", "Speedup", "Efficiency");

    double base_time = 0.0;

    for (int t = 0; t < num_threads; t++) {
        int threads = thread_counts[t];
        double start = omp_get_wtime();

        #pragma omp parallel num_threads(threads)
        {
            #pragma omp single
            {
                #pragma omp task
                vector_add(a, b, add_res, n);

                #pragma omp task
                vector_sub(a, b, sub_res, n);

                #pragma omp task
                vector_mul(a, b, mul_res, n);

                #pragma omp taskwait
            }
        }

        double end = omp_get_wtime();
        double time_taken = end - start;

        if (threads == 1) base_time = time_taken;

        double speedup = base_time / time_taken;
        double efficiency = speedup / threads;

        printf("%10d %15f %15.2f %15.2f\n", threads, time_taken, speedup, efficiency);
    }

    printf("\nSample Results (first 5 elements):\n");
    printf("Add: ");
    for (int i = 0; i < (n < 5 ? n : 5); i++) printf("%d ", add_res[i]);
    printf("\nSub: ");
    for (int i = 0; i < (n < 5 ? n : 5); i++) printf("%d ", sub_res[i]);
    printf("\nMul: ");
    for (int i = 0; i < (n < 5 ? n : 5); i++) printf("%d ", mul_res[i]);
    printf("\n");

    return 0;
}

