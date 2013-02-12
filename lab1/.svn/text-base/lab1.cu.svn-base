//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>


//
// kernel routine
// 

__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  
  //race condition ?

  x[tid - 1] = 2;
  x[tid ] = 1;
}



//
// main code
//

int main(int argc, char **argv)
{
  float *h_x, *d_x;
  int   nblocks, nthreads, nsize, n; 

  // initialise card


  // set number of blocks, and threads per block

  nblocks  = 3000;
  nthreads = 32;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  h_x = (float *)malloc(nsize*sizeof(float));


  cutilSafeCall(cudaMalloc((void **)&d_x, nsize*sizeof(float)));

  // execute kernel

  my_first_kernel<<<nblocks,nthreads>>>(d_x);
  cutilCheckMsg("my_first_kernel execution failed\n");

  cudaThreadSynchronize();


  // copy back results and print them out

  cutilSafeCall( cudaMemcpy(h_x,d_x,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

  cutilSafeCall(cudaFree(d_x));
  free(h_x);

  return 0;
}

 
