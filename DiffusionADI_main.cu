// Import libraries
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <magma_v2.h>
#include <magma_types.h>
#include <magma_lapack.h>

const int numStreams = 1;
const int maxThreadsPerBlock = 1024;

// Store variables into constant memory
__device__ __constant__ int dev_N;
__device__ __constant__ double dev_lambda;
__device__ __constant__ double dev_phi_C;
__device__ __constant__ double dev_alpha;
__device__ __constant__ double dev_beta;

// 
__global__ void init_grid(double* u);
__global__ void init_A(double* A);
__global__ void expl_x(double* u, double* du);
__global__ void expl_y(double* u, double* du);
__global__ void impl_x(double* u, double* du, double* A);
__global__ void transpose(double* u, double* uT);

int main(int argc, char* argv[]) {

  // Read in inputs
  const int N = atol(argv[1]);
  double dx = atof(argv[2]);
  double tau = atof(argv[3]);
  double D = atof(argv[4]);
  double R = atof(argv[5]);
  double C = atof(argv[6]);
  double phi_C = atof(argv[7]);
  int num_iter = atof(argv[8]);
  
  int localN = N/numStreams;

  // Define grid and thread dimensions
  // Turn the following into an if statement later
  // to account for N > 1024
  const int threadsPerBlock = localN;
  const int blocksPerGrid = localN;

  // Define variable for ADI scheme
  double lambda = (tau*D)/(pow(dx,2)*2);
  double time_step = tau/2;
  double alpha = time_step*R;
  double beta = time_step*C;
  printf("%lf\n", lambda);
  
  magma_init();
  magma_int_t *piv, info;
  magma_int_t m = localN;
  magma_int_t n = localN;
  magma_int_t err;
  
  //Declare matrices on host
  double *u, *du, *uT, *A;

  // Declare matrices for device
  double* dev_u[numStreams];
  double* dev_du[numStreams];
  double* dev_uT[numStreams];
  double* dev_A[numStreams];

  // Send varibales to constant memory
  cudaMemcpyToSymbol(dev_N, &N, sizeof(const int));
  cudaMemcpyToSymbol(dev_lambda, &lambda, sizeof(double));
  cudaMemcpyToSymbol(dev_phi_C, &phi_C, sizeof(double));
  cudaMemcpyToSymbol(dev_alpha, &alpha, sizeof(double));
  cudaMemcpyToSymbol(dev_beta, &beta, sizeof(double));

  // Initialize memory on host
  cudaHostAlloc((void**)&u, localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&du, localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&uT, localN*localN*sizeof(double), cudaHostAllocDefault);
  err = magma_dmalloc_cpu(&A, m*m);

  if (err) {
    printf("Issue in memory allocation cpu\n");
    exit(1);
  }

  // Initialize memory on device
  cudaStream_t stream[numStreams];
  for (int i=0; i<numStreams; ++i) {
    cudaMalloc((void**)&dev_u[i], localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_du[i], localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_uT[i], localN*localN*sizeof(double));
    err = magma_dmalloc(&dev_A[i], m*m);

    if (err) {
      printf("Issue in memory allocation gpu\n");
      exit(1);
    }
    
    cudaStreamCreate( &(stream[i]) );
  }

  piv = (magma_int_t*)malloc(m*sizeof(magma_int_t));
  
  /*
  int count = 0;
  // Generate matrix A
  for (int i=0; i < localN; ++i) {
    for (int j=0; j < localN; ++i) {
      if ((i == 0 && j == 0) || (i == localN-1 && j == localN-1)) {
	A[count++] = 1;
      } else if ((i == 0 && j == 1) || (i == (localN-1) && j == (localN-2) 
      } else if (i-j == 0) {
	A[count++] = 1 + 2*lambda;
      } else if (i-j == -1 || i-j == 1) {
	A[count++] = -lambda;
      } else {
	A[count++] = 0;
      }
    }
  }
  */
  
  /*
  for (int i=0; i<(N*N); ++i) {
    if (i % N == 0 || i % N == (N-1)) {
      u[i] = phi_C;
    } else {
      u[i] = 0;
    }
  }
  */
  
  //for (int i=0; i<numStreams; ++i)
  //  cudaMemcpyAsync(dev_u[i], u+i*localN, localN*localN*sizeof(double), cudaMemcpyHostToDevice, stream[i]);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  for (int i=0; i<numStreams; ++i) {
    // Initialize grid using kernel
    init_grid<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(dev_u[i]);
    // Initialize implicit matrix
    init_A<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(dev_A[i]);
    magma_dgetmatrix(m, n, dev_A[i], m, A+i*localN, m, 0);
    magma_dgetrf_gpu(m, m, dev_A[i], m, piv, &info);
    
    for (int j=0; j<num_iter; ++j) {
      //printf("%d,", j);
      // Iterate explicitly in the x direction using kernel
      expl_x<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(dev_u[i], dev_du[i]);
      // Transpose grid in kernel
      transpose<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(dev_du[i], dev_uT[i]);
      // Iterate implicitly in the y direction in kernel
      magma_dgetrs_gpu(MagmaTrans, m, n, dev_A[i], m, piv, dev_uT[i], m, &info);
      expl_x<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(dev_uT[i], dev_du[i]);
      transpose<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(dev_du[i], dev_u[i]);
      magma_dgetrs_gpu(MagmaTrans, m, n, dev_A[i], m, piv, dev_u[i], m, &info);
    }
  }
  
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  
  for (int i=0; i<numStreams; ++i) {
    cudaMemcpyAsync(du+i*localN, dev_du[i], localN*localN*sizeof(double), cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(u+i*localN, dev_u[i], localN*localN*sizeof(double), cudaMemcpyHostToDevice, stream[i]);
  }

  for (int i=0; i<numStreams; ++i)
    cudaStreamSynchronize(stream[i]);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  for (int i = 0; i<numStreams; ++i) {
    cudaFree(dev_u);
    cudaFree(dev_du);
  }
  
  for (int i = 0; i<numStreams; ++i)
    cudaStreamDestroy(stream[i]);
  
  FILE *fileid = fopen("Test_du.csv", "w");

  for (int i=0; i<(N*N); ++i)
    if (i % N == 0) {
      fprintf(fileid, "%lf", du[i]);
    } else if (i % N < (N-1)) {
      fprintf(fileid, ",%lf", du[i]);
    } else {
      fprintf(fileid, ",%lf\n", du[i]);
    }

  fclose(fileid);

  
  FILE *fileid2 = fopen("Test_u.csv", "w");
  
  for (int i=0; i<(N*N); ++i)
    if (i % N == 0) {
      fprintf(fileid2, "%lf", u[i]);
    } else if (i % N < (N-1)) {
      fprintf(fileid2, ",%lf", u[i]);
    } else {
      fprintf(fileid2, ",%lf\n", u[i]);
    }

  fclose(fileid2);

  FILE *fileid3 = fopen("Test_A.csv", "w");
  
  for (int i=0; i<(N*N); ++i)
    if (i % N == 0) {
      fprintf(fileid2, "%lf", A[i]);
    } else if (i % N < (N-1)) {
      fprintf(fileid2, ",%lf", A[i]);
    } else {
      fprintf(fileid2, ",%lf\n", A[i]);
    }

  fclose(fileid3);
  
  cudaFreeHost(u);
  cudaFreeHost(du);

  printf("Kernel Time: %gms\n", elapsedTime);
  
  return 0;
}

__global__ void init_grid(double* u) {
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;

  int g_i = l_i + dev_N*l_j;
  
  if (l_i == 0 || l_j == 0) {
    //printf("left_corner\n");
    u[g_i] = dev_phi_C;
  } else if (l_i == dev_N-1 || l_j == dev_N-1) {
    //printf("right_corner\n");
    u[g_i] = dev_phi_C;
  } else {
    u[g_i] = dev_phi_C;
  }
}

__global__ void init_A(double* A) {
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;

  int g_i = l_i + dev_N*l_j;
  
  if ((l_i == 0 && l_j == 0) || (l_i == dev_N-1 && l_j == dev_N-1)) {
    A[g_i] = 1;
  } else if ((l_i == 1 && l_j == 0) || (l_i == (dev_N-2) && l_j == (dev_N-1))) {
    A[g_i] = 0;
  } else if (l_i-l_j == 0) {
    A[g_i] = 1 + 2*dev_lambda;
  } else if (l_i-l_j == -1 || l_i-l_j == 1) {
    A[g_i] = -dev_lambda;
  } else {
    A[g_i] = 0;
  }

  //printf("%lf\n", A[g_i]);
} 

__global__ void expl_x(double* u, double* du) {
  __shared__ double localu[maxThreadsPerBlock];
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;

  // Index to pull data from global into local
  int g_i = l_i + dev_N*l_j;
  
  localu[l_i] = u[g_i];
  
  __syncthreads();

  double react_term;
  double phi_V = dev_phi_C - 10;

  if (localu[l_i] > phi_V) {
    react_term = -dev_alpha*localu[l_i] - dev_beta;
  } else if (localu[l_i] < phi_V && localu[l_i] > dev_beta) {
    react_term = -dev_beta;
  } else {
    react_term = -localu[l_i];
  }  
  
  if (l_i == 0 || l_i == (dev_N - 1)) {
    du[g_i] = localu[l_i];
  } else if (l_j == 0 || l_j == (dev_N - 1)) {
    du[g_i] = localu[l_i];
  } else {
    //printf("%lf\n", dev_lambda);
    du[g_i] = localu[l_i] + (dev_lambda*localu[l_i + 1] - 2*dev_lambda*localu[l_i] + dev_lambda*localu[l_i-1])
      + react_term;
  }

  //u[g_i] = du[g_i];
}


__global__ void expl_y(double* u, double* du) {
  __shared__ double localu[maxThreadsPerBlock];
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;

  // Transpose coordinates index from global memory to local
  int g_i = l_j + dev_N*l_i;

  localu[l_i] = u[g_i];
  
  __syncthreads();
  
  if (l_i == 0 || l_i == (dev_N - 1)) {
    du[g_i] = localu[l_i];
  } else if (l_j == 0 || l_j == (dev_N-1)) {
    du[g_i] = localu[l_i];
  } else {
    du[g_i] = localu[l_i] + (dev_lambda*localu[l_i + 1] - 2*dev_lambda*localu[l_i] + dev_lambda*localu[l_i-1]);
  }

  //u[g_i] = du[g_i];
}

__global__ void transpose(double* u, double* uT) {
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;

  //int g_i = l_i + dev_N*l_j;

  uT[l_j + dev_N*l_i] = u[l_i + dev_N*l_j];
}
