#include <stdio.h>
#include <mpi.h>

int factorial(int n) {
    int res = 1;
    for (int i = 2; i <= n; i++)
        res *= i;
    return res;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_fact = factorial(rank + 1);
    int scan_sum = 0;

    MPI_Scan(&local_fact, &scan_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("Process %d: partial sum of factorials (1! to %d!) = %d\n", rank, rank+1, scan_sum);

    if (rank == size - 1) {
        printf("Final result (1! + 2! + ... + %d!) = %d\n", size, scan_sum);
    }

    MPI_Finalize();
    return 0;
}
