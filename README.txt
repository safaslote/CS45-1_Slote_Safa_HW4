Safa Slote
CS451 Homework 4

In order to compile matrixNorm.cu, use the following commands:
nvcc -arch=sm_35 -rdc=true matrixNorm.cu -o matrixNorm -lcudadevrt
./matrixNorm <matrix size> <random seed> <number of threads> <number of blocks>


In order to compile matrixNorm.c, use the following commands:
gcc matrixNorm.c -o normSeq
./normSeq <matrix size> <random seed> <number of threads>
