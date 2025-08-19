#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX_LEN 1024

int count_words(char *str) {
    int count = 0, in_word = 0;
    for (int i = 0; str[i]; i++) {
        if (isspace(str[i])) in_word = 0;
        else if (!in_word) { in_word = 1; count++; }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int rank, size, words = 0;
    char line[MAX_LEN];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        FILE *f = fopen("input.txt", "r");
        for (int i = 1; i < size; i++) {
            if (fgets(line, MAX_LEN, f)) {
                line[strcspn(line, "\n")] = 0;
            } else {
                line[0] = 0;
            }
            MPI_Send(line, MAX_LEN, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
        fclose(f);

        int total = 0;
        for (int i = 1; i < size; i++) {
            MPI_Recv(&words, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            printf("Process %d counted %d words\n", i, words);
            total += words;
        }
        printf("Total words = %d\n", total);

    } else {
        MPI_Recv(line, MAX_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        words = count_words(line);
        MPI_Send(&words, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
