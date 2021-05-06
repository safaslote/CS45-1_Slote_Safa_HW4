# CS451_Slote_Safa_HW4
# In order to compile "matrixNorm.cu", use the following commands:
nvcc -arch=sm_35 -rdc=true matrixNorm.cu -o matrixNorm -lcudadevrt
./matrixNorm <matrix size (N)> <random seed> <number of threads> <number of blocks>
