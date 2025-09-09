#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, M;
    #define MAX_M 100
    #define MAX_N 16
    double all_data[MAX_N * MAX_M];
    double my_data[MAX_M];
    double my_avg, total_avg;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter M (number of elements per process, max %d): ", MAX_M);
        scanf("%d", &M);
        if (M > MAX_M) {
            printf("M exceeds MAX_M (%d)\n", MAX_M);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Enter %d elements:\n", size * M);
        for (int i = 0; i < size * M; i++) {
            scanf("%lf", &all_data[i]);
        }
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(all_data, M, MPI_DOUBLE, my_data, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double sum = 0.0;
    for (int i = 0; i < M; i++) sum += my_data[i];
    my_avg = sum / M;
    printf("Process %d of %d: My average = %.2f\n", rank, size, my_avg);

    double all_avgs[MAX_N];
    MPI_Gather(&my_avg, 1, MPI_DOUBLE, all_avgs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        total_avg = 0.0;
        for (int i = 0; i < size; i++) total_avg += all_avgs[i];
        total_avg /= size;
        for (int i = 0; i < size; i++) {
            printf("Process %d average: %.2f\n", i, all_avgs[i]);
        }
        printf("Total average: %.2f\n", total_avg);
    }
    MPI_Finalize();
    return 0;
}
