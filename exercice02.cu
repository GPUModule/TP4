#include <stdio.h>
#include <cuda.h>

__global__ void dummy_kernel(double *data, int N, int idx) {
	int i = blockIdx.x * blockDim.x + blockDim.x*idx + threadIdx.x;
	if (i < N) {
		for (int j = 0; j < 200; j++) {
			data[i] = cos(data[i]);
			data[i] = sqrt(fabs(data[i]));
		}
	}
}

int main()
{
	int nblocks = 30;
	int blocksize = 1024;
	double *data;
	cudaMalloc( (void**)&data, nblocks*blocksize*sizeof(double) );
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	dim3 dimBlock( blocksize, 1, 1 );
	dim3 dimGrid( 1, 1, 1 );
	for (int i = 0; i < nblocks; i++)
		dummy_kernel<<<dimGrid,dimBlock>>>(data, nblocks*blocksize, i);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Temps de l'implementation sequentielle:  %g ms\n", time);

#ifdef USE_STREAMS
	// 2.1 Creation des streams
 
 
	cudaEventRecord(start, 0);
	cudaEventSynchronize(start);
	// 2.2 Execution des kernels
 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Temps de l'implementation parallel:  %g ms\n", time);

	// 2.3 Destruction des streams
 

#endif

	cudaFree( data );
	return EXIT_SUCCESS;
}


