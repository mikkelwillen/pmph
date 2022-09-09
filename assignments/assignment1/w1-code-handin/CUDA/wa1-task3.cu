#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void squareKernel(float* d_in, float* gpu_out) {
    const unsigned int lid = threadIdx.x; // local id inside a block
    const unsigned int gid = blockIdx.x*blockDim.x + lid; // global id
    gpu_out[gid] = pow(d_in[gid]/(d_in[gid] - 2.3), 3.0); // do computation
}

void squareSeq (unsigned int n, float* cpu_out) {
        for (unsigned int i = 0; i < n; i++) {
            cpu_out[i] = pow((float)i/((float)i - 2.3), 3.0);
        }
    }

int main(int argc, char** argv) {
    unsigned int N = atoi(argv[1]);
    unsigned int mem_size = N*sizeof(float);
    unsigned int blocksize = 256;

    // allocate host memory
    float* h_in = (float*) malloc(mem_size);
    float* h_out = (float*) malloc(mem_size);
    float* cpu_out = (float*) malloc(mem_size);

    // initialize the memory
    for (unsigned int i = 0; i < N; ++i) {
        h_in[i] = (float)i;
    }

    // allocate device memory
    float* d_in;
    float* gpu_out;
    
    cudaMalloc((void**)&d_in, mem_size);
    cudaMalloc((void**)&gpu_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    // execute the kernel
    squareKernel<<< (blocksize - 1 + N)/blocksize, blocksize >>>(d_in, gpu_out);

    // execute the seq
    squareSeq(N, cpu_out);

    // copy result from device to host
    cudaMemcpy(h_out, gpu_out, mem_size, cudaMemcpyDeviceToHost);

    // print result
    for (unsigned int i = 0; i < N; ++i) {
        if (cpu_out[i] - h_out[i] > 0.0001) {
            printf("øv bøv\n");
            printf("%-6f - %.6f\n", cpu_out[i], h_out[i]);
        }
    }

    // clean-up memory
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(gpu_out);
}