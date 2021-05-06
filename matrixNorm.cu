#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

#define MAXN 6000  /* Matrix size */
int numThreads; //number of threads
int numBlocks; //number of blocks
int N; //define a max value for N
int row, col;
/* Matrices */
float A[MAXN*MAXN], B[MAXN*MAXN];
//int *ptrA;
float *ptrA = A;
float *ptrB = B;
size_t sizeOfMatrix = sizeof(float) * N *N;
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
	printf("\nNumber of blocks = %i.\n", numBlocks);
        printf("\nNumber of threads = %i.\n", numThreads);
}

/* Initialize A and B*/
void initialize_inputs() {
    int row, col;
    
    srand((unsigned)time(NULL));
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
           A[col*N+row] = (float)rand() / 32768.0;
           B[col*N+row] = 0.0;
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
	printf("%5.2f%s", A[col*N+row], (col < N-1) ? ", " : ";\n\t");
      }
    
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
}
}
}
//using reduction algorithm to calculate first two steps

__global__ void matrixNorm(float *f_A, float *f_B, int n){
     float mu, sigma;
//declare the dimensions x-axis and y-axis (row and col)
     int row  = blockDim.y * blockIdx.y + threadIdx.y;  
     int col = blockDim.x * blockIdx.x + threadIdx.x;
//     int index = col + row * n;
//we want to check to make sure we don't have an excess number of threads
     if(row<n && col<n){
        for(col = 0; col < n; col++){
	   mu = 0.0;
	   for(row = 0; row < n; row++)	
               mu+= f_A[col * n + row];
           mu /= (float) n;
           //you cannot calculate sigma without the mean, so we need some synchronization heree
           //to make sure threads have calculated the mean before getting to this step
           __syncthreads();
           sigma = 0.0;
           for(row = 0; row < n; row++){
	      sigma += powf(f_A[col*n+row] - mu, 2.0);
	   sigma /= (float) n;
           sigma = sqrt(sigma);
           //again, we need to make sure that sigma has been calculated in order to result the normalized matrix
	   __syncthreads();
           for (row = 0; row < n; row++){
               if(sigma == 0.0)
			f_B[row*n+col] = 0.0;
	       else
		        f_B[row*n+col] = (f_A[col*n+row] - mu) / sigma;
		}
	}
}
}
}

int main(int argc, char **argv) {
    /* Timing variables */
    struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    //unsigned long long runtime;
    unsigned long long usecstart, usecstop;
    struct tms cputstart, cputstop;
 
    /* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    initialize_inputs();
    
    /* Print input matrices */
    print_inputs();
    
    /* Start Clock */
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    gettimeofday(&etstart, &tzdummy);
    times(&cputstart);
    
    /* declare arrays */
    
    /* Allocate memory of the matrix in device and copy the matrix from host to device to do work */
    //first, declare size of the matrix
    sizeOfMatrix = N * N * sizeof(float);
    //create pointer to matrix
    float *f_A, *f_B;
    //cuda malloc the matrix (make room for it in memory
    cudaMalloc((void**)&f_A, sizeOfMatrix);
    cudaMalloc((void**)&f_B, sizeOfMatrix);
    
    //initialize start and stop
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //copy data from host to device
    cudaMemcpy(f_A, &ptrA, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    //*m_A = &A[N][N];
    cudaMemcpy(f_B, &ptrB, sizeOfMatrix, cudaMemcpyHostToDevice);
    //use cuda checks to make sure the allocation was successful
    cudaEventRecord(start);
    

    
    /* Matrix Normalization */
    matrixNorm<<<numBlocks, numThreads>>>(f_A, f_B, N);
    cudaError_t err = cudaGetLastError();
    
    if(err != cudaSuccess){
	printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    cudaEventRecord(stop);
    
    /* Free Cuda */
    cudaFree(f_A);
    cudaFree(f_B);
    
    /* Stop Clock */
    //gettimeofday(&stop, &tzdummy);
   // runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);
    
    /* Stop Clock CPU Times */
    gettimeofday(&etstop, &tzdummy);
  times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;  
 
    /* Display timing results */
    //printf("Runtime = %g ms.\n", (float)runtime/(float)1000);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");
    
    /* Display other timing results */
  printf("\nElapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);

  printf("(CPU times are accurate to the nearest %g ms)\n",
	 1.0/(float)CLOCKS_PER_SEC * 1000.0);
  printf("My total CPU time for parent = %g ms.\n",
	 (float)( (cputstop.tms_utime + cputstop.tms_stime) -
		  (cputstart.tms_utime + cputstart.tms_stime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My system CPU time for parent = %g ms.\n",
	 (float)(cputstop.tms_stime - cputstart.tms_stime) /
	 (float)CLOCKS_PER_SEC * 1000);
  printf("My total CPU time for child processes = %g ms.\n",
	 (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
		  (cputstart.tms_cutime + cputstart.tms_cstime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
      /* Contrary to the man pages, this appears not to include the parent */
  printf("--------------------------------------------\n");
    exit(0);
}
