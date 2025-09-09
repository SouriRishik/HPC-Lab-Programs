#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	#define MAXLEN 256
	char S1[MAXLEN] = {0}, S2[MAXLEN] = {0};
	int len = 0;
	int chunk = 0;

	if (rank == 0) {
		printf("Enter String S1: ");
		scanf("%255s", S1);
		printf("Enter String S2: ");
		scanf("%255s", S2);
		len = strlen(S1);
		if (len != (int)strlen(S2)) {
			printf("Error: Strings must be of the same length.\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (len % size != 0) {
			printf("Error: String length must be divisible by number of processes.\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
	chunk = len / size;

	char local_S1[MAXLEN] = {0};
	char local_S2[MAXLEN] = {0};
	char local_result[2*MAXLEN] = {0};

	MPI_Scatter(S1, chunk, MPI_CHAR, local_S1, chunk, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Scatter(S2, chunk, MPI_CHAR, local_S2, chunk, MPI_CHAR, 0, MPI_COMM_WORLD);

	for (int i = 0; i < chunk; i++) {
		local_result[2*i] = local_S1[i];
		local_result[2*i+1] = local_S2[i];
	}

	char result[2*MAXLEN+1] = {0};
	MPI_Gather(local_result, 2*chunk, MPI_CHAR, result, 2*chunk, MPI_CHAR, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		result[2*len] = '\0';
		printf("Resultant String: %s\n", result);
	}

	MPI_Finalize();
	return 0;
}
