#include <mpi.h>
#include <stdio.h>

#define MAX_PROCS 100

int main(int argc, char *argv[]) {
    int rank, size;
    int arr[MAX_PROCS];
    int value, result;
    MPI_Status status;
    char buf[MPI_BSEND_OVERHEAD + sizeof(int)];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Buffer_attach(buf, sizeof(buf));

    if (rank == 0) {
        printf("Enter %d elements:\n", size);
        for (int i = 0; i < size; i++) {
            scanf("%d", &arr[i]);
        }

        for (int i = 1; i < size; i++) {
            MPI_Bsend(&arr[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        if (rank % 2 == 0)
            result = value * value;
        else
            result = value * value * value;

        printf("Process %d received %d, result = %d\n", rank, value, result);
    }

    MPI_Buffer_detach(&buf, &size);
    MPI_Finalize();
    return 0;
}
