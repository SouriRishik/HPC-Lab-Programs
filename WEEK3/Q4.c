#include <stdio.h>
#include <omp.h>

#define MAX_SIZE 100000

int main() {
    int n;
    int arr[MAX_SIZE];

    printf("Enter number of elements (max %d): ", MAX_SIZE);
    scanf("%d", &n);
    if (n <= 0 || n > MAX_SIZE) {
        printf("Invalid size.\n");
        return 1;
    }

    printf("Enter %d elements:\n", n);
    for (int i = 0; i < n; i++)
        scanf("%d", &arr[i]);

    int sum_critical = 0;
    int sum_atomic = 0;
    int sum_reduction = 0;
    int sum_master = 0;
    int sum_locks = 0;

    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel shared(sum_critical)
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            #pragma omp critical
            sum_critical += arr[i];
        }
    }

    #pragma omp parallel shared(sum_atomic)
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            #pragma omp atomic
            sum_atomic += arr[i];
        }
    }

    #pragma omp parallel for reduction(+:sum_reduction)
    for (int i = 0; i < n; i++) {
        sum_reduction += arr[i];
    }

    sum_master = 0;
    #pragma omp parallel
    {
        #pragma omp master
        {
            for (int i = 0; i < n; i++)
                sum_master += arr[i];
        }
    }

    sum_locks = 0;
    #pragma omp parallel shared(sum_locks, lock)
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            omp_set_lock(&lock);
            sum_locks += arr[i];
            omp_unset_lock(&lock);
        }
    }

    omp_destroy_lock(&lock);

    printf("\nSum results using different OpenMP constructs:\n");
    printf("Critical section sum = %d\n", sum_critical);
    printf("Atomic sum           = %d\n", sum_atomic);
    printf("Reduction sum        = %d\n", sum_reduction);
    printf("Master sum           = %d\n", sum_master);
    printf("Locks sum            = %d\n", sum_locks);

    return 0;
}

