#include <stdio.h>
#include <omp.h>

#define MAX_SIZE 100000
#define MAX_THREADS 8

int parallel_sequential_search(int arr[], int n, int target, int num_threads) {
    int index = -1;

    #pragma omp parallel for num_threads(num_threads) shared(index)
    for (int i = 0; i < n; i++) {
        if (index != -1) continue;

        if (arr[i] == target) {
            #pragma omp critical
            {
                if (index == -1 || i < index) {
                    index = i;
                }
            }
        }
    }
    return index;
}

int main() {
    int arr[MAX_SIZE];
    int n, target;
    int thread_counts[] = {1, 2, 4, 8};
    int num_threads = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("Enter array size (max %d): ", MAX_SIZE);
    scanf("%d", &n);
    if (n <= 0 || n > MAX_SIZE) {
        printf("Invalid array size.\n");
        return 1;
    }

    printf("Enter %d elements:\n", n);
    for (int i = 0; i < n; i++)
        scanf("%d", &arr[i]);

    printf("Enter target to search: ");
    scanf("%d", &target);

    printf("\nParallel Sequential Search Performance (OpenMP)\n");
    printf("%10s %15s %15s %15s\n", "Threads", "Time (s)", "Speedup", "Efficiency");

    double base_time = 0.0;

    for (int t = 0; t < num_threads; t++) {
        int threads = thread_counts[t];

        double start = omp_get_wtime();
        int found_index = parallel_sequential_search(arr, n, target, threads);
        double end = omp_get_wtime();

        double time_taken = end - start;

        if (threads == 1) base_time = time_taken;

        double speedup = base_time / time_taken;
        double efficiency = speedup / threads;

        printf("%10d %15f %15.2f %15.2f\n", threads, time_taken, speedup, efficiency);

        if (found_index == -1)
            printf("Target not found in array.\n");
        else
            printf("Target found at index: %d\n", found_index);
    }

    return 0;
}

