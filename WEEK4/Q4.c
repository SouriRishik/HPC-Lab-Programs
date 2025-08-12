#include <mpi.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

void toggle_char(char *c)
{
    if (islower(*c))
        *c = toupper(*c);
    else if (isupper(*c))
        *c = tolower(*c);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char str[] = "HeLLO";
    int len = strlen(str);

    if (size != len)
    {
        if (rank == 0)
        {
            printf("Number of processes (%d) must be equal to string length (%d).\n", size, len);
        }
        MPI_Finalize();
        return 1;
    }
    char ch = str[rank];
    toggle_char(&ch);
    printf("Process %d toggled character '%c' to '%c'\n", rank, str[rank], ch);
    MPI_Finalize();
    return 0;
}
