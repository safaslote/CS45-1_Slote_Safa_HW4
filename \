#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

#define N 6000  /* Matrix size */
int numThreads; //number of threads
int numBlocks; //number of blocks
int MAXN = 9000; //define a max value for N
/* Matrices */
volatile float A[N][N], B[N][N];

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
	int seed = 0;  /* Random seed */
	// char uid[32]; /*User name */

	/* Read command-line arguments */
	srand(time_seed());  /* Randomize */

	//changed count to 5 so that user can insert number of grid blocks
	if (argc == 5) {
		seed = atoi(argv[2]);
		srand(seed);
		numThreads=atoi(argv[3]);
		numBlocks=atoi(argv[4]); //insert argument for number of blocks
		printf("Random seed = %i\n", seed);
	} 
	if (argc >= 2) {
		N = atoi(argv[1]);
		if (N < 1 || N > MAXN) {
			printf("N = %i is out of range.\n", N);
			exit(0);
		}
	}
	else {
		printf("Usage: %s <matrix_dimension> [random seed]\n",
				argv[0]);    
		exit(0);
	}

	/* Print parameters */
	printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B*/
void initialize_inputs() {
    int row, col;
    
    srand((unsigned)time(NULL));
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row][col] = (float)rand() / 32768.0;
            B[row][col] = 0.0;
        }
    }
    
}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
}
}
//using reduction algorithm to calculate first two steps

__global__ void matrixNorm(float *f_A, float *f_B, int n){
     float mu, sigma;
//declare the dimensions x-axis and y-axis (row and col)
     int row  = blockDim.y * blockIdx.y + threadIdx.y;  
     int col = blockDim.x * blockIdx.x + threadIdx.x;
     int index = col + row * n;
//we want to check to make sure we don't have an excess number of threads
     if(row<N && col<N){
        for(col = 0; col < N; col++){
	   mu = 0.0;
	   for(row = 0; row < N; row++)	
               mu+= f_A[index];
           mu /= (float) N;
           //you cannot calculate sigma without the mean, so we need some synchronization heree
           //to make sure threads have calculated the mean before getting to this step
           cudaThreadSynchronize();
           sigma = 0.0;
           for(row = 0; row < N; row++){
	      sigma += powf(f_A[index] - mu, 2.0);
	   sigma /= (float) N;
           sigma = sqrt(sigma);
           //again, we need to make sure that sigma has been calculated in order to result the normalized matrix
	   cudaThreadSynchronize();
           for (row = 0; row < N; row++){
               if(sigma == 0.0)
	       else
		        f_B[index] = (f_A[index] - mu) / sigma;
		}
	}
}
}


int main(int argc, char **argv) {
    /* Timing variables */
    struct timeval start, stop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    unsigned long long runtime;
    
    /* Initialize A and B */
    initialize_inputs();
    
    
    /* Start Clock */
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    gettimeofday(&start, &tzdummy);
    
    /* declare arrays */
    
    /* Allocate memory of the matrix in device and copy the matrix from host to device to do work */
    //first, declare size of the matrix
    size_t sizeOfMatrix = N * N * sizeof(float);
    //create pointer to matrix
    float (*f_A)[N], (*f_B)[N];
    //cuda malloc the matrix (make room for it in memory
    cMallocA = cudaMalloc((void**)&f_A, sizeOfMatrix);
    cMallocB = cudaMalloc((void**)&f_B, sizeOfMatric);
    //copy data from host to device
    cudaMemcpy(f_A, A, sizeOfMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(f_B, B, sizeOfMatrix, cudaMemcpyHostToDevice);
    //use cuda checks to make sure the allocation was successful
    
    

    
    /* Matrix Normalization */
    matrixNorm<<<numBlocks, numThreads>>>(f_A, f_B, N);
    
    
    /* Free Cuda */
    cudaFree(f_A);
    cudaFree(f_B);
    
    /* Stop Clock */
    gettimeofday(&stop, &tzdummy);
    runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);
    
    
    /* Display timing results */
    printf("Runtime = %g ms.\n", (float)runtime/(float)1000);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");
    
    exit(0);
}
