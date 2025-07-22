#include <stdio.h>
#include <omp.h>

int reverseDigits(int num) {
    int rev = 0;
    while (num > 0) {
        rev = rev * 10 + num % 10;
        num /= 10;
    }
    return rev;
}

int main() {
    int input[9] = {18, 523, 301, 1234, 2, 14, 108, 150, 1928};
    int output[9];
    int i;
    
    #pragma omp parallel for
    for (i = 0; i < 9; i++) {
    	int tid = omp_get_thread_num();
    	printf("omp thread %d\n", tid);
        output[i] = reverseDigits(input[i]);
    }

    printf("Output array: ");
    for (i = 0; i < 9; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    return 0;
}
