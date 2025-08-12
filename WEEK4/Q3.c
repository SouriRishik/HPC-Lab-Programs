#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    int rank, size;
    int num1 = 12, num2 = 4;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
    {
        float add = num1 + num2;
        printf("Process %d: Addition: %.2f\n", rank, add);
    }
    else if (rank == 1)
    {
        float sub = num1 - num2;
        printf("Process %d: Subtraction: %.2f\n", rank, sub);
    }
    else if (rank == 2)
    {
        float mul = num1 * num2;
        printf("Process %d: Multiplication: %.2f\n", rank, mul);
    }
    else
    {
        if (num2 != 0)
        {
            float div = num1 / num2;
            printf("Process %d: Division: %.2f\n", rank, div);
        }
        else
        {
            printf("Process %d: Division by zero is not possible",rank);
        }
    }
    MPI_Finalize();
    return 0;
}
