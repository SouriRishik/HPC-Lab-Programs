#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <ctype.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Status status;
    char word[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "This program requires at least 2 MPI processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {

        strcpy(word, "HelloMPI"); 
        printf("Process 0: Sending word '%s' to Process 1\n", word);
        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        MPI_Ssend(word, sizeof(word), MPI_CHAR, 1, 1, MPI_COMM_WORLD);
        MPI_Recv(word, sizeof(word), MPI_CHAR, 1, 1, MPI_COMM_WORLD, &status);
        printf("Process 0: Received toggled word '%s' from Process 1\n", word);

    } else if (rank == 1) {

        MPI_Recv(word, sizeof(word), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process 1: Received word '%s' from Process 0\n", word);

        for (int i = 0; i < strlen(word); i++) {
            if (islower(word[i])) {
                word[i] = toupper(word[i]);
            } else if (isupper(word[i])) {
                word[i] = tolower(word[i]);
            }
        }

        printf("Process 1: Sending toggled word '%s' back to Process 0\n", word);
        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}