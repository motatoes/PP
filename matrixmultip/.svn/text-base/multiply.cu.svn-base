#include "stdio.h"
#include "cutil_inline.h"
#include "math.h"

// YOU MAY WANT TO CHANGE THIS
#define BLOCK_SIZE_X 2
#define BLOCK_SIZE_Y 2
#define GRID_SIZE_X 2
#define GRID_SIZE_Y 4

// DO NOT CHANGE THESE
#define A_WIDTH 8
#define A_HEIGHT 4
#define B_WIDTH 8


#define ITERS 100


void datainit(float*,int);
void datainit(float*,int, int);

__global__ void axb_in_c(float* a, float* b, float* c)
{
	int t_x = threadIdx.x;
	int t_y = threadIdx.y;
	int b_x = blockIdx.x * BLOCK_SIZE_X;
	int a_y = blockIdx.y * BLOCK_SIZE_Y;

	// c[b_x+t_x][a_y+t_y] = 0.0;
	c[ (b_x+t_x)*B_WIDTH + a_y+t_y] = 0;

	for(int a_x = 0; a_x < A_WIDTH; a_x += 1)
	{
		int b_y = a_x;

		// c[b_x+t_x][a_y+t_y] += a[a_x][a_y+t_y]*b[b_x+t_x][b_y]
		c[(b_x + t_x) * B_WIDTH + (a_y+t_y)  ] +=  a[ (b_x+t_x) * A_WIDTH + (a_x)]
							    * b[ b_y  * B_WIDTH + (b_x + t_x)];
		__syncthreads();

	}

}

int main(int argc, char** argv)
{
  int devID;
  cudaDeviceProp props;

  // get number of SMs on this GPU
  cutilSafeCall(cudaGetDevice(&devID));
  cutilSafeCall(cudaGetDeviceProperties(&props, devID));

  // allocate host memory 
  unsigned int size_a     = A_WIDTH * A_HEIGHT;              
  unsigned int size_b	  = B_WIDTH * A_WIDTH; //A_WIDTH = B_HEIGHT
  unsigned int size_c     = A_HEIGHT * B_WIDTH;

  unsigned int mem_size_a = sizeof(float) * size_a;    
  unsigned int mem_size_b = sizeof(float) * size_b;     
  unsigned int mem_size_c = sizeof(float) * size_c;     
  
  float* h_A     =   (float*)malloc(mem_size_a);    
  float* h_B     =   (float*)malloc(mem_size_b); 
  float* h_C     =   (float*)malloc(mem_size_c);

  //printf("Input size : %d\n", VECTOR_SIZE);
  printf("Grid size_X  : %d\n", GRID_SIZE_X);
  printf("Grid size_Y  : %d\n", GRID_SIZE_Y);
  printf("Block size_X : %d\n", BLOCK_SIZE_X);
  printf("Block size_Y : %d\n", BLOCK_SIZE_Y);

  // initialize A and B on the host
  datainit(h_A, size_a,1);
  datainit(h_B, size_b,2);
  //datainit(h_C, size_c, -1);
  // allocate device memory
  float* d_A;
  cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_a));
  float* d_B;
  cutilSafeCall(cudaMalloc((void**) &d_B, mem_size_b));  
  float* d_C;
  cutilSafeCall(cudaMalloc((void**) &d_C, mem_size_c));

  // copy host memory to device
  cutilSafeCall(cudaMemcpy(d_A, h_A, mem_size_a, cudaMemcpyHostToDevice)); 
  cutilSafeCall(cudaMemcpy(d_B, h_B, mem_size_b, cudaMemcpyHostToDevice));

  // set up kernel for execution
  printf("Run %d Kernels.\n\n", ITERS);
  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));  

  //int nsteps = log(VECTOR_SIZE) / log(2);
  //printf("nsteps ----- %d \n", nsteps);
  
  dim3 grid_size;
  grid_size.x = GRID_SIZE_X;
  grid_size.y = GRID_SIZE_Y;

  // create two dimensional 16x16 thread blocks
  dim3 block_size;
  block_size.x = BLOCK_SIZE_X;
  block_size.y = BLOCK_SIZE_Y;


// execute kernel
  //for (int j = 0; j < ITERS; j++) 
      axb_in_c<<<grid_size, block_size >>>(d_A, d_B, d_C);

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");

  // wait for device to finishA_HEIGHT
  cudaThreadSynchronize();

  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  double dSeconds = cutGetTimerValue(timer)/(1000.0); 
  double dNumOps = ITERS * (size_a+size_b+size_c);
  double gflops = dNumOps/dSeconds/1.0e9;

  //Log througput
  printf("Throughput = %.4f GFlop/s\n", gflops);
  cutilCheckError(cutDeleteTimer(timer));

  // copy result from device to host
  cutilSafeCall(cudaMemcpy(h_C, d_C, 
				       mem_size_c, cudaMemcpyDeviceToHost));

  // error check
  printf("C = \n");
  for (int i=0 ; i < A_HEIGHT ; i++)
  {
	for (int j=0 ; j < B_WIDTH ; j++)
  	  {
  		printf("%.4f   ", h_C[ i * B_WIDTH +j]);
	  }
  	printf("\n");
  }

  // clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  cutilSafeCall(cudaFree(d_A));
  cutilSafeCall(cudaFree(d_B));
  cutilSafeCall(cudaFree(d_C));

  // exit and clean up device status
  cudaThreadExit();
}

// Allocates a matrix with random float entries.
void datainit(float* data, int size)
{
  srand(time(NULL));
  for (int i = 0; i < size; ++i)
    data[i] = 1;//rand()/(float)RAND_MAX;
}


void datainit(float* data, int size, int constant)
{
  srand(time(NULL));
  for (int i = 0; i < size; ++i)
    data[i] = constant;//rand()/(float)RAND_MAX;
}


