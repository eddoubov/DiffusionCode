// Import libraries
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <magma_v2.h>
#include <magma_types.h>
#include <magma_lapack.h>
#include <math.h>

const int numStreams = 1;
const int maxThreadsPerBlock = 1024;
const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

// Store variables into constant memory
__device__ __constant__ int dev_N;
__device__ __constant__ double dev_lambda;
__device__ __constant__ double dev_phi_C;
__device__ __constant__ double dev_alpha;
__device__ __constant__ double dev_beta;

//
__global__ void test_func();
//__global__ void init_grid(double* u, int update);
//__global__ void init_A(double* A);
__global__ void expl_x(double* u, double* du, double dt);
__global__ void expl_y(double* u, double* du, double dt);
__global__ void expl_z(double* u, double* du, double dt);
__global__ void transpose(double* u, double* uT, int dir);
__global__ void comb_u(double* u1, double* u2, double* u3, int pom);
__global__ void add_react_term(double* u, double* uN, int mod_bool);
__global__ void copy(double* u1, double* u2);

void swap (int *a, int *b) {
  int temp = *a;
  //printf("%d\n", *a);
  *a = *b;
  *b = temp;
}

void randomize (int array[], int n, long int seed) {
  
  for (int i = n-1; i > 0; i--) {
    int j = lrand48() % (i+1);

    if (i == 10) {
      printf("%d\n", j);
    }

    swap(&array[i], &array[j]);
  }
}

void get_coords (int index, int localN, int* x_coord, int* y_coord, int* z_coord) {

  int a = index;
  *z_coord = a/(localN*localN);
  int b = a % (localN*localN);
  *y_coord = b/(localN);
  int c = b % localN;
  *x_coord = c;
  
}


int main(int argc, char* argv[]) {

  // Read in inputs
  const int N = atol(argv[1]);
  double dx = atof(argv[2]);
  double tau = atof(argv[3]);
  double D = atof(argv[4]);
  double R = atof(argv[5]);
  double C = atof(argv[6]);
  double phi_C = atof(argv[7]);
  double cap_vf = atof(argv[8]);
  int num_iter = atol(argv[9]);

  long int seed;
  if (argc > 10) {
    seed = atol(argv[100]);
  } else {

    FILE* urand = fopen("/dev/urandom", "r");

    fread( &seed, sizeof(long int), 1, urand);

    fclose(urand);
  }

  int localN = N/numStreams;
  int gridsize = localN*localN*localN;

  long int Vt = N*N*N;
  long int Nc = Vt*cap_vf;
  double mu = N/cbrt(Nc);
  int sigma = 60/dx;

  printf("%ld\n", Nc);
  printf("%lf\n", mu);

  // Define grid and thread dimensions
  // Turn the following into an if statement later
  // to account for N > 1024
  //const int tPB2D = localN;
  //const int bPG2D = localN;

  // Define variable for ADI scheme
  double lambda = (tau*D)/(pow(dx,2));
  double time_step = tau;
  double alpha = time_step*R;
  double beta = time_step*C;
  printf("%lf\n", lambda);
  printf("%lf\n", alpha);

  // Start Monte-Carlo Method
  srand48(seed);
  printf("%ld\n", seed);
  
  int *rand_indices, *cap_indices;
  cudaHostAlloc((void**)&rand_indices, gridsize*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void**)&cap_indices, gridsize*sizeof(int), cudaHostAllocDefault);

  for (int i = 0; i < gridsize; ++i) {
    rand_indices[i] = i;
  }

  printf("%d\n", rand_indices[10]);
  
  randomize(rand_indices, gridsize, seed);

  printf("%d\n", rand_indices[10]);
  
  int cap_count = 0;
  int x_coord, y_coord, z_coord;
  int x_cap, y_cap, z_cap;
  int temp_index, cap_index;
  
  for (int i=0; i< gridsize; ++i) {

    temp_index = rand_indices[i];
    
    get_coords(temp_index, localN, &x_coord, &y_coord, &z_coord);
    
    if (i == 0) {
      cap_indices[cap_count] = temp_index;
      cap_count++;
      continue;
    }

    double min_dist_cap;
    
    for (int j=0; j<cap_count; ++j) {
      cap_index = cap_indices[j];
      get_coords(cap_index, localN, &x_cap, &y_cap, &z_cap);

      double expr = (x_coord-x_cap)*(x_coord-x_cap) + (y_coord-y_cap)*(y_coord-y_cap)
	+ (z_coord-z_cap)*(z_coord-z_cap);
      double dist = sqrt(expr);

      if (j==0) {
	min_dist_cap = dist;
      } else if(dist < min_dist_cap) {
	min_dist_cap = dist;
      }
    }    

    if (i == 10) {
      printf("%d,", x_cap);
      printf("%d,", y_cap);
      printf("%d\n", z_cap);
    }
    
  }
  
  
  magma_init();
  magma_int_t *piv, info;
  magma_int_t m = localN;
  magma_int_t n = localN;
  magma_int_t err;
  
  //Declare matrices on host
  double *u, *du, *uT, *uN, *A;
  //double *u1, *u2, *u3, *u4, *u5, *u6, *u7, *u8, *u9, *u10;

  // Declare matrices for device
  double* dev_u[numStreams];
  double* dev_du[numStreams];
  double* dev_uT[numStreams];
  double* dev_uN[numStreams];
  magmaDouble_ptr dev_A;

  /**
  double* dev_u1[numStreams];
  double* dev_u2[numStreams];
  double* dev_u3[numStreams];
  double* dev_u4[numStreams];
  double* dev_u5[numStreams];
  double* dev_u6[numStreams];
  double* dev_u7[numStreams];
  double* dev_u8[numStreams];
  double* dev_u9[numStreams];
  double* dev_u10[numStreams];
  **/
  
  // Send varibales to constant memory
  cudaMemcpyToSymbol(dev_N, &N, sizeof(const int));
  cudaMemcpyToSymbol(dev_lambda, &lambda, sizeof(double));
  cudaMemcpyToSymbol(dev_phi_C, &phi_C, sizeof(double));
  cudaMemcpyToSymbol(dev_alpha, &alpha, sizeof(double));
  cudaMemcpyToSymbol(dev_beta, &beta, sizeof(double));

  // Initialize memory on host
  cudaHostAlloc((void**)&u, gridsize*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&du, gridsize*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&uT, gridsize*sizeof(double), cudaHostAllocDefault);  //delete later
  cudaHostAlloc((void**)&uN, gridsize*sizeof(double), cudaHostAllocDefault);  //delete later

  /**
  cudaHostAlloc((void**)&u1, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&u2, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&u3, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&u4, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&u5, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&u6, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&u7, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&u8, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&u9, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&u10, localN*localN*localN*sizeof(double), cudaHostAllocDefault);
  **/
  
  err = magma_dmalloc_cpu(&A, m*m);

  if (err) {
    printf("Issue in memory allocation cpu: A\n");
    exit(1);
  }

  // Initialize memory on device
  cudaStream_t stream[numStreams];
  for (int i=0; i<numStreams; ++i) {
    cudaMalloc((void**)&dev_u[i], gridsize*sizeof(double));
    cudaMalloc((void**)&dev_du[i], gridsize*sizeof(double));
    cudaMalloc((void**)&dev_uT[i], gridsize*sizeof(double));
    cudaMalloc((void**)&dev_uN[i], gridsize*sizeof(double));

    /**
    cudaMalloc((void**)&dev_u1[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_u2[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_u3[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_u4[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_u5[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_u6[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_u7[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_u8[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_u9[i], localN*localN*localN*sizeof(double));
    cudaMalloc((void**)&dev_u10[i], localN*localN*localN*sizeof(double));
    **/
    
    err = magma_dmalloc(&dev_A, m*m);
    if (err) {
      printf("Issue in memory allocation gpu\n");
      exit(1);
    }
    
    cudaStreamCreate( &(stream[i]) );
  }

  piv = (magma_int_t*)malloc(m*sizeof(magma_int_t));

  int ind;
  
  for (int j=0; j<localN; ++j) {
    for (int i=0; i<localN; ++i) {
      ind = i + j*localN;
      if ((i == 0 && j == 0) || (i == localN-1 && j == localN-1)) {
	A[ind] = 1;
      } else if ((i == 1 && j == 0) || (i == (localN-2) && j == (localN-1))) {
	A[ind] = 0;
      } else if (i-j == 0) {
	A[ind] = 1 + lambda;
      } else if (i-j == -1 || i-j == 1) {
	A[ind] = -lambda/2;
      } else {
	A[ind] = 0;
      }
    }
  }

  magma_dsetmatrix(m, m, A, m, dev_A, m, 0);

  int update = 0;
  
  for (int k=0; k<localN; ++k) {
    for (int j=0; j<localN; ++j) {
      for (int i=0; i<localN; ++i) {
	ind  = i + j*localN + k*localN*localN;

	if (i == 0 || j == 0 || k == 0) {
	  //printf("left_corner\n");
	  u[ind] = phi_C;
	} else if (i == localN-1 || j == localN-1 || k == localN-1) {
	  //printf("right_corner\n");
	  u[ind] = phi_C;
	} else if (update == 0) {
	  u[ind] = phi_C;
	}
      }
    }
  }
  
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);  

  dim3 bPG3D(localN,localN);
  //dim3 block(threadsPerBlock, threadsPerBlock, threadsPerBlock);
  dim3 tPB3D(localN);
  dim3 bPG2D_trans(localN/TILE_DIM, localN/TILE_DIM, localN);
  dim3 tPB2D_trans(TILE_DIM, BLOCK_ROWS, 1);
  
  for (int i=0; i<numStreams; ++i) {
    // Initialize grid using kernel
    //init_grid<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], 0);
    // Initialize implicit matrix
    //init_A<<<bPG2D, tPB2D, 0, stream[i]>>>(dev_A[i]);

    magma_dgetmatrix(m, n, dev_A, m, A+i*localN, m, 0);
    magma_dgetrf_gpu(m, m, dev_A, m, piv, &info); 
    
    for (int j=0; j<num_iter; ++j) {
      // Iterate explicitly in the x direction using kernel
      //test_func<<<grid, block>>>();
      cudaMemcpyAsync(dev_u[i], u+i*localN, gridsize*sizeof(double), cudaMemcpyHostToDevice, stream[i]);

      if (j > 0) {
	cudaMemcpyAsync(dev_du[i], du+i*localN, gridsize*sizeof(double), cudaMemcpyHostToDevice, stream[i]);
	cudaMemcpyAsync(dev_uN[i], uN+i*localN, gridsize*sizeof(double), cudaMemcpyHostToDevice, stream[i]);
      }
      
      add_react_term<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_uN[i], 0);
      comb_u<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_u[i], dev_uN[i], dev_uN[i], 1);
      //copy<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_u[i], dev_u1[i]);  // copy for debugging
      expl_x<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_du[i], 2);
      comb_u<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_du[i], dev_uN[i], 1);
      //copy<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_u1[i]);  // copy for debugging
      expl_y<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_du[i], 1);
      comb_u<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_du[i], dev_uN[i], 1);
      //copy<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_u2[i]);  //copy for debugging
      expl_z<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_du[i], 1);
      comb_u<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_du[i], dev_uN[i], 1);
      //copy<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_u3[i]);  //copy for debugging
      magma_dgetmatrix(m, n, dev_A, m, A+i*localN, m, 0);
      magma_dgetrs_gpu(MagmaTrans, m, n*n, dev_A, m, piv, dev_uN[i], m, &info);
      //copy<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_u4[i]);  //copy for debugging
      add_react_term<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_uN[i], dev_uT[i], 0);
      comb_u<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_uT[i], dev_uN[i], 1);
      expl_y<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_du[i], 2);
      comb_u<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_du[i], dev_uN[i], 0);
      //copy<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_u5[i]);  //copy for debugging
      transpose<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_uT[i], 1);
      //copy<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uT[i], dev_u6[i]);  //copy for debugging
      magma_dgetmatrix(m, n, dev_A, m, A+i*localN, m, 0);
      magma_dgetrs_gpu(MagmaTrans, m, n*n, dev_A, m, piv, dev_uT[i], m, &info);
      //copy<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uT[i], dev_u7[i]);  //copy for debugging
      transpose<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uT[i], dev_uN[i], 1);
      add_react_term<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_uN[i], dev_uT[i], 0);
      comb_u<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_uT[i], dev_uN[i], 1);
      expl_z<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], dev_du[i], 2);
      comb_u<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_du[i], dev_uN[i], 0);
      //copy<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_u8[i]);  //copy for debugging
      transpose<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uN[i], dev_uT[i], 0);
      magma_dgetmatrix(m, n, dev_A, m, A+i*localN, m, 0);
      //copy<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uT[i], dev_u9[i]);  //copy for debugging
      magma_dgetrs_gpu(MagmaTrans, m, n*n, dev_A, m, piv, dev_uT[i], m, &info);
      transpose<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_uT[i], dev_u[i], 0);
      //copy<<<bPG2D_trans, tPB2D_trans, 0, stream[i]>>>(dev_u[i], dev_u10[i]);  //copy for debugging
      //init_grid<<<bPG3D, tPB3D, 0, stream[i]>>>(dev_u[i], 1);
      cudaMemcpyAsync(u+i*localN, dev_u[i], gridsize*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
      cudaMemcpyAsync(du+i*localN, dev_du[i], gridsize*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
      cudaMemcpyAsync(uN+i*localN, dev_uN[i], gridsize*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
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

  /**
  for (int i=0; i<numStreams; ++i) {
    //cudaMemcpyAsync(du+i*localN, dev_du[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    //cudaMemcpyAsync(u+i*localN, dev_u[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync(u1+i*localN, dev_u1[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync(u2+i*localN, dev_u2[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync(u3+i*localN, dev_u3[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync(u4+i*localN, dev_u4[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync(u5+i*localN, dev_u5[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync(u6+i*localN, dev_u6[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync(u7+i*localN, dev_u7[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync(u8+i*localN, dev_u8[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync(u9+i*localN, dev_u9[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync(u10+i*localN, dev_u10[i], localN*localN*localN*sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
  }
  **/

  for (int i=0; i<numStreams; ++i)
    cudaStreamSynchronize(stream[i]);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  for (int i = 0; i<numStreams; ++i) {
    cudaFree(dev_u[i]);
    cudaFree(dev_du[i]);
    cudaFree(dev_uT[i]);
    cudaFree(dev_uN[i]);

    /**
    cudaFree(dev_u1[i]);
    cudaFree(dev_u2[i]);
    cudaFree(dev_u3[i]);
    cudaFree(dev_u4[i]);
    cudaFree(dev_u5[i]);
    cudaFree(dev_u6[i]);
    cudaFree(dev_u7[i]);
    cudaFree(dev_u8[i]);
    cudaFree(dev_u9[i]);
    cudaFree(dev_u10[i]);
    **/
  }

  //printf("cuda_test: %lf\n", u1[17]);
  
  for (int i = 0; i<numStreams; ++i)
    cudaStreamDestroy(stream[i]);

  //dgetri(m, A, n, piv
  
  char fn1[25], fn2[23];
  //char fn3[25], fn4[25], fn5[25], fn6[25], fn7[25], fn8[25], fn9[25], fn10[25], fn11[25], fn12[27];

  for (int j=0; j<N; ++j) {

    int offset = j*N*N;

    sprintf(fn1, "du_files/Test_du_%d.csv", j);
    sprintf(fn2, "u_files/Test_u_%d.csv", j);
    /**
    sprintf(fn3, "u1_files/Test_u1_%d.csv", j);
    sprintf(fn4, "u2_files/Test_u2_%d.csv", j);
    sprintf(fn5, "u3_files/Test_u3_%d.csv", j);
    sprintf(fn6, "u4_files/Test_u4_%d.csv", j);
    sprintf(fn7, "u5_files/Test_u5_%d.csv", j);
    sprintf(fn8, "u6_files/Test_u6_%d.csv", j);
    sprintf(fn9, "u7_files/Test_u7_%d.csv", j);
    sprintf(fn10, "u8_files/Test_u8_%d.csv", j);
    sprintf(fn11, "u9_files/Test_u9_%d.csv", j);
    sprintf(fn12, "u10_files/Test_u10_%d.csv", j);
    **/
    
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

    /**
    FILE *fileid3 = fopen(fn3, "w");
  
    for (int i=0; i<(N*N); ++i) {
      //printf("%d\n", i);
      //printf("%lf\n", u1[i+offset]);
      if (i % N == 0) {
	fprintf(fileid3, "%lf", u1[i+offset]);
      } else if (i % N < (N-1)) {
	fprintf(fileid3, ",%lf", u1[i+offset]);
      } else {
	fprintf(fileid3, ",%lf\n", u1[i+offset]);
      }
    }

    fclose(fileid3);

    
    FILE *fileid4 = fopen(fn4, "w");
  
    for (int i=0; i<(N*N); ++i) {
      if (i % N == 0) {
	fprintf(fileid4, "%lf", u2[i+offset]);
      } else if (i % N < (N-1)) {
	fprintf(fileid4, ",%lf", u2[i+offset]);
      } else {
	fprintf(fileid4, ",%lf\n", u2[i+offset]);
      }
    }

    fclose(fileid4);

    FILE *fileid5 = fopen(fn5, "w");
  
    for (int i=0; i<(N*N); ++i) {
      if (i % N == 0) {
	fprintf(fileid5, "%lf", u3[i+offset]);
      } else if (i % N < (N-1)) {
	fprintf(fileid5, ",%lf", u3[i+offset]);
      } else {
	fprintf(fileid5, ",%lf\n", u3[i+offset]);
      }
    }

    fclose(fileid5);

    FILE *fileid6 = fopen(fn6, "w");
  
    for (int i=0; i<(N*N); ++i) {
      if (i % N == 0) {
	fprintf(fileid6, "%lf", u4[i+offset]);
      } else if (i % N < (N-1)) {
	fprintf(fileid6, ",%lf", u4[i+offset]);
      } else {
	fprintf(fileid6, ",%lf\n", u4[i+offset]);
      }
    }

    fclose(fileid6);

    FILE *fileid7 = fopen(fn7, "w");
  
    for (int i=0; i<(N*N); ++i) {
      if (i % N == 0) {
	fprintf(fileid7, "%lf", u5[i+offset]);
      } else if (i % N < (N-1)) {
	fprintf(fileid7, ",%lf", u5[i+offset]);
      } else {
	fprintf(fileid7, ",%lf\n", u5[i+offset]);
      }
    }

    fclose(fileid7);

    FILE *fileid8 = fopen(fn8, "w");
  
    for (int i=0; i<(N*N); ++i) {
      if (i % N == 0) {
	fprintf(fileid8, "%lf", u6[i+offset]);
      } else if (i % N < (N-1)) {
	fprintf(fileid8, ",%lf", u6[i+offset]);
      } else {
	fprintf(fileid8, ",%lf\n", u6[i+offset]);
      }
    }

    fclose(fileid8);

    FILE *fileid9 = fopen(fn9, "w");
  
    for (int i=0; i<(N*N); ++i) {
      if (i % N == 0) {
	fprintf(fileid9, "%lf", u7[i+offset]);
      } else if (i % N < (N-1)) {
	fprintf(fileid9, ",%lf", u7[i+offset]);
      } else {
	fprintf(fileid9, ",%lf\n", u7[i+offset]);
      }
    }

    fclose(fileid9);

    FILE *fileid10 = fopen(fn10, "w");
  
    for (int i=0; i<(N*N); ++i) {
      if (i % N == 0) {
	fprintf(fileid10, "%lf", u8[i+offset]);
      } else if (i % N < (N-1)) {
	fprintf(fileid10, ",%lf", u8[i+offset]);
      } else {
	fprintf(fileid10, ",%lf\n", u8[i+offset]);
      }
    }

    fclose(fileid10);

    FILE *fileid11 = fopen(fn11, "w");
  
    for (int i=0; i<(N*N); ++i) {
      if (i % N == 0) {
	fprintf(fileid11, "%lf", u9[i+offset]);
      } else if (i % N < (N-1)) {
	fprintf(fileid11, ",%lf", u9[i+offset]);
      } else {
	fprintf(fileid11, ",%lf\n", u9[i+offset]);
      }
    }

    fclose(fileid11);

    FILE *fileid12 = fopen(fn12, "w");
  
    for (int i=0; i<(N*N); ++i) {
      if (i % N == 0) {
	fprintf(fileid12, "%lf", u10[i+offset]);
	} else if (i % N < (N-1)) {
	fprintf(fileid12, ",%lf", u10[i+offset]);
	} else {
	fprintf(fileid12, ",%lf\n", u10[i+offset]);
	}
      }

    fclose(fileid12);
    **/
  }

  FILE *fileid13 = fopen("Test_A.csv", "w");
  
  for (int i=0; i<(N*N); ++i)
    if (i % N == 0) {
      fprintf(fileid13, "%lf", A[i]);
    } else if (i % N < (N-1)) {
      fprintf(fileid13, ",%lf", A[i]);
    } else {
      fprintf(fileid13, ",%lf\n", A[i]);
    }

  fclose(fileid13);
    
  free(A);
  magma_free(dev_A);
  free(piv);
  magma_finalize();
  
  cudaFreeHost(u);
  cudaFreeHost(du);
  cudaFreeHost(uN);
  cudaFreeHost(uT);

  /**
  cudaFreeHost(u1);
  cudaFreeHost(u2);
  cudaFreeHost(u3);
  cudaFreeHost(u4);
  cudaFreeHost(u5);
  cudaFreeHost(u6);
  cudaFreeHost(u7);
  cudaFreeHost(u8);
  cudaFreeHost(u9);
  cudaFreeHost(u10);
  **/
  
  printf("Kernel Time: %gms\n", elapsedTime);
  
  return 0;
}

  /**
__global__ void test_func() {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  printf("GPU - i = %d, j = %d, k = %d\n", i, j, k);
  printf("Threadx = %d, Blockx = %d, Thready = %d, Blocky = %d\n", threadIdx.x, blockIdx.x, threadIdx.y, blockIdx.y);
  
}
  **/

/**
__global__ void init_grid(double* u, int update) {
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
  } else if (update == 0) {
    u[g_i] = dev_phi_C;
  }
}
**/

/**
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
**/

__global__ void expl_x(double* u, double* du, double dt) {
  __shared__ double localu[maxThreadsPerBlock];
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;
  int l_k = blockIdx.y;

  // Index to pull data from global into local
  int g_i = l_i + dev_N*l_j + dev_N*dev_N*l_k;
  
  localu[l_i] = u[g_i];
  
  __syncthreads();

  double temp_lambda = dev_lambda/dt;
  
  if (l_i == 0 || l_i == (dev_N - 1)) {
    du[g_i] = 0;
  } else if (l_j == 0 || l_j == (dev_N - 1)) {
    du[g_i] = 0;
  } else if (l_k == 0 || l_k == (dev_N - 1)) {
    du[g_i] = 0;
  } else {
    du[g_i] = (temp_lambda*localu[l_i+1] - 2*temp_lambda*localu[l_i] + temp_lambda*localu[l_i-1]);
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

  double temp_lambda = dev_lambda/dt;
  
  if (l_i == 0 || l_i == (dev_N - 1)) {
    du[g_i] = 0;
  } else if (l_j == 0 || l_j == (dev_N - 1)) {
    du[g_i] = 0;
  } else if (l_k == 0 || l_k == (dev_N - 1)) {
    du[g_i] = 0;
  } else {
    du[g_i] = (temp_lambda*localu[l_i+1] - 2*temp_lambda*localu[l_i] + temp_lambda*localu[l_i-1]);
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

  double temp_lambda = dev_lambda/dt;
  
  if (l_i == 0 || l_i == (dev_N - 1)) {
    du[g_i] = 0;
  } else if (l_j == 0 || l_j == (dev_N - 1)) {
    du[g_i] = 0;
  } else if (l_k == 0 || l_k == (dev_N - 1)) {
    du[g_i] = 0;
  } else {
    du[g_i] = (temp_lambda*localu[l_i+1] - 2*temp_lambda*localu[l_i] + temp_lambda*localu[l_i-1]);
  }
  
}

__global__ void transpose(double* idata, double* odata, int dir) {
  __shared__ double localu[TILE_DIM][TILE_DIM];

  if (dir == 1) {

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int z = blockIdx.z * dev_N * dev_N;
    int width = gridDim.x * TILE_DIM;

    //printf("%d\n", gridDim.x);
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      
      localu[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x + z];
      
    }
    
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      
      odata[(y+j)*width + x + z] = localu[threadIdx.x][threadIdx.y + j];
      
    }

  } else if (dir == 0) {

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int z = blockIdx.y * TILE_DIM + threadIdx.y;
    int y = blockIdx.z * dev_N;
    int width = gridDim.x * TILE_DIM * dev_N;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      
      localu[threadIdx.y+j][threadIdx.x] = idata[(z+j)*width + x + y];
      
    }
    
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    z = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      
      odata[(z+j)*width + x + y] = localu[threadIdx.x][threadIdx.y + j];
      
    }
  }
}

__global__ void comb_u(double* mat1, double* mat2, double* mat3, int pom) {
  /**
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;
  int l_k = blockIdx.y;

  int g_i = l_i + dev_N*l_j + dev_N*dev_N*l_k;
  **/
  
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int z = blockIdx.z * dev_N * dev_N;
  int width = gridDim.x * TILE_DIM;
  
  if (pom == 1) {
    //mat3[g_i] = mat1[g_i] + mat2[g_i];
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
      mat3[(y+j)*width + x + z] = mat1[(y+j)*width + x + z] + mat2[(y+j)*width + x + z];
  } else {
    //mat3[g_i] = mat1[g_i] - mat2[g_i];
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
      mat3[(y+j)*width + x + z] = mat1[(y+j)*width + x + z] - mat2[(y+j)*width + x + z];
  }
}

__global__ void add_react_term(double* idata, double* odata, int mod_bool) {
  int l_i = threadIdx.x;
  int l_j = blockIdx.x;
  int l_k = blockIdx.y;

  int g_i = l_i + dev_N*l_j + dev_N*dev_N*l_k;

  double react_term;
  double phi_V = dev_phi_C - 10;
  
  if (mod_bool == 1) {
    if (idata[g_i] > phi_V) {
      react_term = -dev_alpha*idata[g_i]/2;
    } else {
      react_term = 0;
    }
  } else {
    if (idata[g_i] > phi_V) {
      react_term = -dev_alpha*idata[g_i]/2 - dev_beta/2;
    } else if (idata[g_i] < phi_V && idata[g_i] > dev_beta) {
      react_term = -dev_beta/2;
    } else {
      react_term = -idata[g_i];
    }
  }

  if (l_i == 0 || l_i == (dev_N - 1)) {
    odata[g_i] = 0;
  } else if (l_j == 0 || l_j == (dev_N - 1)) {
    odata[g_i] = 0;
  } else if (l_k == 0 || l_k == (dev_N - 1)) {
    odata[g_i] = 0;
  } else {
    odata[g_i] = react_term;
  }
  
}

__global__ void copy(double* cdata1, double* cdata2) {
  //int l_i = threadIdx.x;
  //int l_j = blockIdx.x;
  //int l_k = blockIdx.y;

  //int g_i = l_i + dev_N*l_j + dev_N*dev_N*l_k;
  
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int z = blockIdx.z * dev_N * dev_N;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    cdata2[(y+j)*width + x + z] = cdata1[(y+j)*width + x + z];
  
  /**
  if (l_k == 0 && l_j == 0 && l_i == 0) {
    printf("%lf\n", u1[g_i]);
  }
  **/
  
  //odata[g_i] = idata[g_i]; 
}

