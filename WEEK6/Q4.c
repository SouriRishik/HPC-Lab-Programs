#include <stdio.h>
#include <mpi.h>

#define N 3

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[N][N];
    int element, local_count = 0, total_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0) {
            printf("This program must be run with 3 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Enter 3x3 matrix (row-wise):\n");
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                scanf("%d", &matrix[i][j]);
        printf("Enter element to search: ");
        scanf("%d", &element);
    }

    MPI_Bcast(&element, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int row[N];
    MPI_Scatter(matrix, N, MPI_INT, row, N, MPI_INT, 0, MPI_COMM_WORLD);

    for (int j = 0; j < N; j++) {
        if (row[j] == element) local_count++;
    }

    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Element %d occurs %d times in the matrix.\n", element, total_count);
    }

    MPI_Finalize();
    return 0;
}
