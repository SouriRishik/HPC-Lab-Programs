#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[4][4];
    int output[4][4];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        printf("Enter the matrix elements for 4x4:\n");
        for(int i=0; i<4; i++) {
            for(int j=0; j<4; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
        printf("Input matrix:\n");
        for(int i=0; i<4; i++) {
            for(int j=0; j<4; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }
    }
    MPI_Bcast(matrix, 16, MPI_INT, 0, MPI_COMM_WORLD);

    int row[4];
    for(int j=0; j<4; j++) {
        if(rank == 0) row[j] = matrix[0][j];
        if(rank == 1) row[j] = matrix[0][j] + matrix[1][j];
        if(rank == 2) row[j] = matrix[0][j] + matrix[2][j];
        if(rank == 3) row[j] = matrix[0][j] + matrix[1][j] + matrix[2][j] + matrix[3][j];
    }
    
    MPI_Gather(row, 4, MPI_INT, output, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("Output matrix:\n");
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                printf("%d ", output[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
