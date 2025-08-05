#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_SIZE 10000
#define MAX_THREADS 4

void selection_sort_parallel(int arr[], int n, int num_threads) {
    for (int i = 0; i < n - 1; i++) {
        int min_index = i;

        #pragma omp parallel for num_threads(num_threads)
        for (int j = i + 1; j < n; j++) {
            #pragma omp critical
            {
                if (arr[j] < arr[min_index]) {
                    min_index = j;
                }
            }
        }

        if (min_index != i) {
            int temp = arr[min_index];
            arr[min_index] = arr[i];
            arr[i] = temp;
        }
    }
}

void copy_array(int *src, int *dest, int n) {
    for (int i = 0; i < n; i++)
        dest[i] = src[i];
}

int main() {
    int original[MAX_SIZE], copy[MAX_SIZE];
    int n;

    printf("Enter number of elements (max %d): ", MAX_SIZE);
    scanf("%d", &n);

    if (n > MAX_SIZE || n <= 0) {
        printf("Invalid size.\n");
        return 1;
    }

    printf("Enter %d array elements:\n", n);
    for (int i = 0; i < n; i++)
        scanf("%d", &original[i]);

    int thread_counts[MAX_THREADS] = {1, 2, 3, 4};

    printf("\nSelection Sort Performance using OpenMP\n");
    printf("%10s %15s %15s %15s\n", "Threads", "Time (s)", "Speedup", "Efficiency");

    double base_time = 0.0;

    for (int t = 0; t < MAX_THREADS; t++) {
        int threads = thread_counts[t];
        copy_array(original, copy, n);

        double start = omp_get_wtime();
        selection_sort_parallel(copy, n, threads);
        double end = omp_get_wtime();

        double time_taken = end - start;

        if (threads == 1)
            base_time = time_taken;

        double speedup = base_time / time_taken;
        double efficiency = speedup / threads;

        printf("%10d %15f %15.2f %15.2f\n", threads, time_taken, speedup, efficiency);
    }

    return 0;
}

