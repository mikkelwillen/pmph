#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void squareKernel(float* d_in, float *d_out) {
    const unsigned int lid = threadIdx.x; // local id inside a block
    const unsigned int gid = blockIdx.x*blockDim.x + lid; // global id
    d_out[gid] = pow((d_in[gid]/(d_in[gid] - 2.3)), 3.0); // do computation
}

int main(int argc, char** argv) {
    unsigned int N = atoi(argv[1]);
    unsigned int mem_size = N*sizeof(float);

    // allocate host memory
    float* h_in = (float*) malloc(mem_size);
    float* h_out = (float*) malloc(mem_size);

    // initialize the memory
    for (unsigned int i = 0; i < N; ++i) {
        h_in[i] = (float)i;
    }

    // allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in, mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    // execute the kernel
    squareKernel<<< (256 - 1 + N)/256, 256>>>(d_in, d_out);

    // copy result from device to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    // print result
    for (unsigned int i = 0; i < N; ++i) {
        printf("%d - %.6f\n", i, h_out[i]);
    }

    // clean-up memory
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
}