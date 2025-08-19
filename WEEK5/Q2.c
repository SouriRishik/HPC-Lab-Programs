#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    int data;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Master process (rank %d) sending numbers to slaves...\n", rank);
        for (int i = 1; i < size; i++) {
            data = i * 10;
            MPI_Send(&data, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            printf("Master sent %d to slave %d\n", data, i);
        }
    } else {
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Slave process (rank %d) received: %d\n", rank, data);
    }
    MPI_Finalize();

    return 0;
}