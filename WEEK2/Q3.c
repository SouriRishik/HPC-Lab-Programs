#include <stdio.h>
#include <omp.h>

int main() {
    int n, i;
    printf("Enter the size of the array: ");
    scanf("%d", &n);

    int arr[n];
    printf("Enter %d elements:\n", n);
    for(i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    int even_sum = 0, odd_sum = 0;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for(int j = 0; j < n; j++) {
                if(arr[j] % 2 == 0)
                    even_sum += arr[j];
            }
            printf("Thread %d calculated even sum: %d\n", omp_get_thread_num(), even_sum);
        }

        #pragma omp section
        {
            for(int j = 0; j < n; j++) {
                if(arr[j] % 2 != 0)
                    odd_sum += arr[j];
            }
            printf("Thread %d calculated odd sum: %d\n", omp_get_thread_num(), odd_sum);
        }
    }

    printf("\nTotal sum of even numbers: %d\n", even_sum);
    printf("Total sum of odd numbers: %d\n", odd_sum);

    return 0;
}