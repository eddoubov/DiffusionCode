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
__device__ __constant__ double dev_tau;
__device__ __constant__ double dev_lambda;
__device__ __constant__ double dev_phi_C;
__device__ __constant__ double dev_alpha;
__device__ __constant__ double dev_beta;

//
__global__ void test_func();
__global__ void init_grid(double* u);
__global__ void init_A(double* A);
__global__ void init_A2(double* A);
__global__ void expl_x(double* u, double* du, double dt);
__global__ void expl_y(double* u, double* du, double dt);
__global__ void expl_z(double* u, double* du, double dt);
__global__ void impl_x(double* u, double* du, double* A);
__global__ void transpose(double* u, double* uT, int dir);
__global__ void comb_u(double* u1, double* u2, double* u3, int pom);

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
  const int tPB2D = localN;
  const int bPG2D = localN;

  // Define variable for ADI scheme
  double lambda = (tau*D)/(pow(dx,2));
  double time_step = tau;
  double alpha = time_step*R;
  double beta = time_step*C;
  //printf("%lf\n", lambda);
  
  magma_init();
  magma_int_t *piv, info;
  magma_int_t m = localN;
  magma_int_t n = localN;
  magma_int_t err;
  magma_int_t err2;
  
  //Declare matrices on host
  double *u, *du, *uT, *uN, *A, *A2;

  // Declare matrices for device
  double* dev_u[numStreams];
  double* dev_du[numStreams];
  double* dev_uT[numStreams];
  double* dev_uN[numStreams];
  double* dev_A[numStreams];
  double* dev_A2[numStreams];

  // Send varibales to constant memory
  cudaMemcpyToSymbol(dev_N, &N, sizeof(const int));
  cudaMemcpyToSymbol(dev_tau, &tau, sizeof(double));
  cudaMemcpyToSymbol(dev_lambda, &lambda, sizeof(double));
  cudaMemcpyToSymbol(dev_phi_C, &phi_C, sizeof(double));
  cudaMemcpyToSymbol(dev_alpha, &alpha, sizeof(double));
  cudaMemcpyToSymbol(dev_beta, &beta, sizeof(double));

  // Initialize memory on host
  cudaHostAlloc((void**)&u, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&du, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&uT, localN*localN*localN*sizeof(double), cudaHostAllocDefault);  //delete later
  cudaHostAlloc((void**)&uN, localN*localN*localN*sizeof(double), cudaHostAllocDefault);  //delete later
  err = magma_dmalloc_cpu(&A, m*m);
  err2 = magma_dmalloc_cpu(&A2, m*m);

  if (err) {
    printf("Issue in memory allocation cpu: A\n");
    exit(1);
  }

  if (err2) {
    printf("Issue in memory allocation cpu: A2\n");
    exit(1)
  }

  // Initialize memory on device
  cudaStream_t stream[numStreams];
  for (int i=0; i<numStreams; ++i) {
    cudaMalloc((void**)&dev_u[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_du[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_uT[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_uN[i], localN*localN*localN*sizeof(double));
    err = magma_dmalloc(&dev_A[i], m*m);
    if (err) {
      printf("Issue in memory allocation gpu\n");
      exit(1);
    }
    err = magma_dmalloc(&dev_A2[i], m*m);
    if (err) {
      printf("Issue in memory allocation gpu\n");
      exit(1);
    }
    
    cudaStreamCreate( &(stream[i]) );
  }

  piv = (magma_int_t*)malloc(m*sizeof(magma_int_t));  

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);  

  dim3 bPG3D(localN,localN);
  //dim3 block(threadsPerBlock, threadsPerBlock, threadsPerBlock);
  dim3 tPB3D(localN);
  
  for (int i=0; i<numStreams; ++i) {
    // Initialize grid using kernel
    init_grid<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i]);
    // Initialize implicit matrix
    init_A<<<bPG2D, tPB2D, 0, stream[i]>>>(dev_A[i]);
    magma_dgetmatrix(m, n, dev_A[i], m, A+i*localN, m, 0);
    magma_dgetrf_gpu(m, m, dev_A[i], m, piv, &info);
    
    for (int j=0; j<num_iter; ++j) {
      // Iterate explicitly in the x direction using kernel
      //test_func<<<grid, block>>>();
      add_react_term<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_uN[i], 0);
      comb_u<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_uN[i], dev_uN[i], 1);
      expl_x<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_du[i], 2);
      comb_u<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_uN[i], dev_du[i], dev_uN[i], 1);
      expl_y<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_du[i], 1);
      comb_u<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_uN[i], dev_du[i], dev_uN[i], 1);
      expl_z<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_du[i], 1);
      comb_u<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_uN[i], dev_du[i], dev_uN[i], 1);
      transpose<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_uN[i], dev_uT[i], 0);
      magma_dgetrs_gpu(MagmaTrans, m, n*n, dev_A[i], m, piv, dev_uT[i], m, &info);
      transpose<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_uT[i], dev_uN[i], 1);
      expl_y<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_du[i], 2);
      comb_u<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_uN[i], dev_du[i], dev_uN[i], 0);
      magma_dgetrs_gpu(MagmaTrans, m, n*n, dev_A[i], m, piv, dev_uN[i], m, &info);
      expl_z<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_du[i], 2);
      comb_u<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_uN[i], dev_du[i], dev_uN[i], 0);
      add_react_term<<<bPG3D, tPB3D, 0 stream[i]>>>(dev_u[i], dev_uT[i], 1);
      comb_u<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_uN[i], dev_uT[i], dev_uN[i], 0);
      
      
      // Transpose grid in kernel
      //transpose<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(dev_du[i], dev_uT[i]);
      // Iterate implicitly in the y direction in kernel
      //magma_dgetrs_gpu(MagmaTrans, m, n, dev_A[i], m, piv, dev_uT[i], m, &info);
      //expl_x<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(dev_uT[i], dev_du[i]);
      //transpose<<<blocksPerGrid, threadsPerBlock, 0, stream[i]>>>(dev_du[i], dev_u[i]);
      //magma_dgetrs_gpu(MagmaTrans, m, n, dev_A[i], m, piv, dev_u[i], m, &info);
    }
  }
  
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  
  for (int i=0; i<numStreams; ++i) {
    cudaMemcpyAsync(du+i*localN, dev_du[i], localN*localN*localN*sizeof(double), cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(u+i*localN, dev_u[i], localN*localN*localN*sizeof(double), cudaMemcpyHostToDevice, stream[i]);
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

  char fn1[25], fn2[23];

  for (int j=0; j<N; ++j) {

    int offset = j*N*N;

    sprintf(fn1, "du_files/Test_du_%d.csv", j);
    sprintf(fn2, "u_files/Test_u_%d.csv", j);

    FILE *fileid = fopen(fn1, "w");

    for (int i=0; i<(N*N); ++i) {
      if (i % N == 0) {
	fprintf(fileid, "%lf", du[i+offset]);
      } else if (i % N < (N-1)) {
	fprintf(fileid, ",%lf", du[i+offset]);
      } else {
	fprintf(fileid, ",%lf\n", du[i+offset]);
      }
    }

    fclose(fileid);
  
    FILE *fileid2 = fopen(fn2, "w");
  
    for (int i=0; i<(N*N); ++i) {
      //printf("%lf\n", u[i+offset]);
      //printf("%d\n", offset);
      if (i % N == 0) {
	fprintf(fileid2, "%lf", u[i+offset]);
      } else if (i % N < (N-1)) {
	fprintf(fileid2, ",%lf", u[i+offset]);
      } else {
	fprintf(fileid2, ",%lf\n", u[i+offset]);
      }
    }

    fclose(fileid2);

  }

  FILE *fileid3 = fopen("Test_A.csv", "w");
  
  for (int i=0; i<(N*N); ++i)
    if (i % N == 0) {
      fprintf(fileid3, "%lf", A[i]);
    } else if (i % N < (N-1)) {
      fprintf(fileid3, ",%lf", A[i]);
    } else {
      fprintf(fileid3, ",%lf\n", A[i]);
    }

  fclose(fileid3);
  
  cudaFreeHost(u);
  cudaFreeHost(du);

  printf("Kernel Time: %gms\n", elapsedTime);
  
  return 0;
}

__global__ void test_func() {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  printf("GPU - i = %d, j = %d, k = %d\n", i, j, k);
  printf("Threadx = %d, Blockx = %d, Thready = %d, Blocky = %d\n", threadIdx.x, blockIdx.x, threadIdx.y, blockIdx.y);
  
}

__global__ void init_grid(double* u) {
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;
  int l_k = blockIdx.y;

  int g_i = l_i + dev_N*l_j + dev_N*dev_N*l_k;
  
  if (l_i == 0 || l_j == 0 || l_k == 0) {
    //printf("left_corner\n");
    u[g_i] = dev_phi_C;
  } else if (l_i == dev_N-1 || l_j == dev_N-1 || l_k == dev_N-1) {
    //printf("right_corner\n");
    u[g_i] = dev_phi_C;
  } else {
    u[g_i] = 0;
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
    A[g_i] = 1 + dev_lambda;
  } else if (l_i-l_j == -1 || l_i-l_j == 1) {
    A[g_i] = -dev_lambda/2;
  } else {
    A[g_i] = 0;
  }
}

__global__ void init_A2(double* A) {
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;

  int g_i = l_i + dev_N*l_j;
  
  if ((l_i == 0 && l_j == 0) || (l_i == dev_N-1 && l_j == dev_N-1)) {
    A[g_i] = 1;
  } else if ((l_i == 1 && l_j == 0) || (l_i == (dev_N-2) && l_j == (dev_N-1))) {
    A[g_i] = 0;
  } else if (l_i-l_j == 0) {
    A[g_i] = 1 + dev_lambda;
  } else if (l_i-l_j == -1 || l_i-l_j == 1) {
    A[g_i] = -dev_lambda/2;
  } else {
    A[g_i] = 0;
  }
}

__global__ void expl_x(double* u, double* du, double dt) {
  __shared__ double localu[maxThreadsPerBlock];
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;
  int l_k = blockIdx.y;

  // Index to pull data from global into local
  int g_i = l_i + dev_N*l_j + dev_N*dev_N*l_k;
  
  localu[l_i] = u[g_i];
  
  __syncthreads();

  double react_term;
  double phi_V = dev_phi_C - 10;
  double temp_lambda = dev_lambda/dt;
  
  if (l_i == 0 || l_i == (dev_N - 1)) {
    du[g_i] = 0
  } else if (l_j == 0 || l_j == (dev_N - 1)) {
    du[g_i] = 0
  } else {
    du[g_i] = (temp_lambda*localu[l_i + 1] - 2*temp_lambda*localu[l_i] + temp_lambda*localu[l_i-1]);
  }
}


__global__ void expl_y(double* u, double* du, double dt) {
  __shared__ double localu[maxThreadsPerBlock];

  int l_i = threadIdx.x;
  int l_j = blockIdx.x;
  int l_k = blockIdx.y;

  // Index to pull data from global into local
  int g_i = l_j + dev_N*l_i + dev_N*dev_N*l_k;
  
  localu[l_i] = u[g_i];
  
  __syncthreads();

  double react_term;
  double phi_V = dev_phi_C - 10;
  double temp_lambda = dev_lambda/dt;
  
  if (l_i == 0 || l_i == (dev_N - 1)) {
    du[g_i] = 0;
  } else if (l_j == 0 || l_j == (dev_N - 1)) {
    du[g_i] = 0;
  } else {
    du[g_i] = (temp_lambda*localu[l_i + 1] - 2*temp_lambda*localu[l_i] + temp_lambda*localu[l_i-1]);
  }
}

__global__ void expl_z(double* u, double* du, double dt) {
  __shared__ double localu[maxThreadsPerBlock];

  int l_i = threadIdx.x;
  int l_j = blockIdx.x;
  int l_k = blockIdx.y;

  // Index to pull data from global into local
  int g_i = l_k + dev_N*l_j + dev_N*dev_N*l_i;
  
  localu[l_i] = u[g_i];
  
  __syncthreads();

  double react_term;
  double phi_V = dev_phi_C - 10;
  double temp_lambda = dev_lambda/dt;
  
  if (l_i == 0 || l_i == (dev_N - 1)) {
    du[g_i] = 0;
  } else if (l_j == 0 || l_j == (dev_N - 1)) {
    du[g_i] = 0;
  } else {
    du[g_i] = (temp_lambda*localu[l_i + 1] - 2*temp_lambda*localu[l_i] + temp_lambda*localu[l_i-1]);
  }
  
}

__global__ void transpose(double* u, double* uT, int dir) {
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;
  int l_k = blockIdx.y;

  if (dir == 1) {
    uT[l_j + dev_N*l_k + dev_N*dev_N*l_i] = u[l_i + dev_N*l_j + dev_N*dev_N*l_k];
  } else {
    uT[l_k + dev_N*l_i + dev_N*dev_N*l_j] = u[l_i + dev_N*l_j + dev_N*dev_N*l_k];
  }
}

__global__ void comb_u(double* u1, double* u2, double* u3, int pom) {
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;
  int l_k = blockIdx.y;

  int g_i = l_i + dev_N*l_j + dev_N*dev_N*l_k;

  if (pom == 1) {
    u3[g_i] = u1[g_i] + u2[g_i];
  } else {
    u3[g_i] = u1[g_i] - u2[g_i];
  }
}

__global__ void add_react_term(double* u, double* uN, int mod_bool) {
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;
  int l_k = blockIdx.y;

  int g_i = l_i + dev_N*l_j + dev_N*dev_N*l_k;

  if (mod_bool == 1) {
    if (u[g_i] > phi_V) {
      react_term = -dev_alpha*u[g_i]/2;
     else {
      react_term = 0;
    }
  } else {
    if (u[g_i] > phi_V) {
      react_term = -dev_alpha*u[g_i] - dev_beta;
    } else if (u[g_i] < phi_V && u[g_i] > dev_beta) {
      react_term = -dev_beta;
    } else {
      react_term = -u[g_i];
    }
  }

  uN[g_i] = react_term;
  
}

