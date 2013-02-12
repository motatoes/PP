
/* Matrix scalar multiplication: B = k * A.
 *
 * This is a naive implementation which uses threads and blocks.
 * It is intended to provide an introduction to CUDA program structure. 
 */

// Utilities and system includes
#include "cutil_inline.h"

/* the kernel
 *
 * Scalar multiplication on the device: B = k* A 
 */

__global__ void
scalarMultGbl( float* B, float* A, int bsize, float k)
{
  // identify thread and block
  int tx = threadIdx.x;
  int bk = blockIdx.x;

  // each thread operates on one element
  B[bk * bsize + tx] = k * A[bk * bsize + tx];
}

bool doThreading(int n);

/* declaration, forward */
void randomInit(float*, int);

/*
 * Program main
 */

int main(int argc, char** argv)
{

  unsigned int nblocks = 1;
  
  while (doThreading(nblocks)) {
	nblocks +=  5;
  }

  // exit and clean up device status
  cudaThreadExit();
}
bool doThreading(int n) {
  
  int nthreads = n;
  bool returnval = true;
  int devID;
  cudaDeviceProp props;

  // get number of SMs on this GPU
  cutilSafeCall(cudaGetDevice(&devID));
  cutilSafeCall(cudaGetDeviceProperties(&props, devID));

  //printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

  if (nthreads > props.maxThreadsPerBlock){
    nthreads = props.maxThreadsPerBlock;
    returnval = false;
  }
 

  // allocate host memory for matrices A and B
  unsigned int bsize = nthreads;//props.maxThreadsPerBlock; // threads per block
  unsigned int size =   props.maxGridSize[0] * bsize;//
  unsigned int gsize = size / bsize;              // total amount of threads needed in the kernel

  unsigned int mem_size = sizeof(float) * size;  // calculate physical array size
  float* h_A = (float*)malloc(mem_size);         // allocate A on host
  float* h_B = (float*)malloc(mem_size);         // allocate B (result) on host

  //printf("Using vector size : %.5f M\n", size/1000000.0); 
  //printf("Grid size         : %d\n", gsize);
  //printf("Block size        : %d\n", bsize);

  // initialize host memory
  srand(2006);                                   // seed random number generator
  randomInit(h_A, size);

  // allocate device memory
  float* d_A;
  cutilSafeCall(cudaMalloc((void**) &d_A, mem_size));  // allocate A on device
  float* d_B;
  cutilSafeCall(cudaMalloc((void**) &d_B, mem_size));  // allocate B on device

  // copy host memory to device
  cutilSafeCall(cudaMemcpy(d_A, h_A, mem_size,
			   cudaMemcpyHostToDevice) );  // copy A from host to device

  // set up kernel for execution
  int nIter = 100;                                     // runt he kernel 100 times
  float k = 1.23;                                      // the scalar
  //printf("\n=== Run %d Kernels. === \n\n", nIter);   
  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));             // create a timer
  cutilCheckError(cutStartTimer(timer));               // start the timer

  // execute the kernel
  for (int j = 0; j < nIter; j++) 
{
      scalarMultGbl<<< gsize, bsize >>>(d_B, d_A, bsize, k);
}

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");

  // wait for device to finish
  cudaThreadSynchronize();

  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  double dSeconds = cutGetTimerValue(timer)/(1000.0);
  double dNumOps = 1.0e-9 * nIter * size;
  double gflops = dNumOps/dSeconds;

  //Log througput, etc

  printf("%.4f,,%.4d\n",gflops,nthreads);
  //printf("Throughput = %.4f GFlops\nTime = %.5f s\nSize = %.5f Gops\n\nBye!\n", 
  //	   gflops, dSeconds, dNumOps);
  cutilCheckError(cutDeleteTimer(timer));

  // copy result from device to host
  cutilSafeCall(cudaMemcpy(h_B, d_B, mem_size,
			   cudaMemcpyDeviceToHost) );

  // clean up memory
  free(h_A);
  free(h_B);
  cutilSafeCall(cudaFree(d_A));
  cutilSafeCall(cudaFree(d_B));


  return returnval;
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
  for (int i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

