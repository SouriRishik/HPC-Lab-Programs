#include <stdio.h>
#include <omp.h>

int main() {
    int start, end;
    int num_threads;
    long long sum = 0;

    printf("Enter interval start and end (start <= end): ");
    scanf("%d %d", &start, &end);

    if (start > end) {
        printf("Invalid interval.\n");
        return 1;
    }

    int n = end - start + 1;

    printf("Enter number of threads: ");
    scanf("%d", &num_threads);
    if (num_threads <= 0) {
        printf("Invalid number of threads.\n");
        return 1;
    }

    long long sums[4];    
    double times[4];      

    const char *schedules[] = {"static", "dynamic", "guided", "auto"};

    double base_time = 0.0;

    for (int i = 0; i < 4; i++) {
        sum = 0;
        double start_time = omp_get_wtime();

        #pragma omp parallel for num_threads(num_threads) schedule(runtime) reduction(+:sum)
        for (int j = start; j <= end; j++) {
            sum += j;
        }

        double end_time = omp_get_wtime();

        times[i] = end_time - start_time;
        sums[i] = sum;

        if (i == 0) base_time = times[0];
    }

    printf("\nSummation results and timings:\n");
    printf("%10s %20s %15s %15s\n", "Schedule", "Sum", "Time (s)", "Speedup");

    for (int i = 0; i < 4; i++) {
        double speedup = base_time / times[i];
        printf("%10s %20lld %15f %15.2f\n", schedules[i], sums[i], times[i], speedup);
    }

    return 0;
}

