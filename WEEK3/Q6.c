#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    long long num_points;
    long long inside_count = 0;

    printf("Enter number of random points: ");
    scanf("%lld", &num_points);

    if (num_points <= 0) {
        printf("Number of points must be positive.\n");
        return 1;
    }

    double pi_estimate = 0.0;

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)(omp_get_thread_num() + 1);

        #pragma omp for reduction(+:inside_count)
        for (long long i = 0; i < num_points; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;

            if (x * x + y * y <= 1.0) {
                inside_count++;
            }
        }
    }

    pi_estimate = 4.0 * ((double)inside_count / (double)num_points);

    printf("Estimated value of pi = %f\n", pi_estimate);

    return 0;
}

