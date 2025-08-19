#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter an integer value: ");
        fflush(stdout);
        scanf("%d", &value);
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&value, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, &status);
        printf("Root process received final value: %d\n", value);

    } else {
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        value++;
        int next = (rank == size - 1) ? 0 : rank + 1;
        MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
