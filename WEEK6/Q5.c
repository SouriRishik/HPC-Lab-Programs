#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, err;
    MPI_Comm err_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm_dup(MPI_COMM_WORLD, &err_comm);
    MPI_Comm_set_errhandler(err_comm, MPI_ERRORS_RETURN);

    int sendbuf = rank;
    int recvbuf;

    int invalid_rank = size + 1;

    err = MPI_Send(&sendbuf, 1, MPI_INT, invalid_rank, 0, err_comm);

    if (err != MPI_SUCCESS) {
        char err_string[MPI_MAX_ERROR_STRING];
        int err_len;
        MPI_Error_string(err, err_string, &err_len);
        printf("Process %d caught MPI error: %s\n", rank, err_string);
    }

    MPI_Comm_free(&err_comm);
    MPI_Finalize();
    return 0;
}
