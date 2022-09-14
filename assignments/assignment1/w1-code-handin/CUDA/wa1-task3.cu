#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void squareKernel(float* d_in, float* gpu_out) {
    const unsigned int lid = threadIdx.x; // local id inside a block
    const unsigned int gid = blockIdx.x * blockDim.x + lid; // global id
    float temp = d_in[gid];
    float inner = temp / (temp - 2.3);
    gpu_out[gid] = inner * inner * inner; // do computation
}

void squareSeq (unsigned int n, float* cpu_out) {
        for (unsigned int i = 0; i < n; i++) {
            cpu_out[i] = pow((float)i/((float)i - 2.3), 3.0);
        }
    }

int timevalSubstract (struct timeval* result, 
                       struct timeval* t2, 
                       struct timeval* t1) {
    unsigned int resolution = 1000000;
    long int diff = (t2 -> tv_usec + resolution * t2 -> tv_sec) - 
                    (t1 -> tv_usec + resolution * t1 -> tv_sec);
    result -> tv_sec  = diff / resolution;
    result -> tv_usec = diff % resolution;
    return (diff<0);
}

#define GPU_RUNS 100
int main(int argc, char** argv) {
    // initial values
    unsigned int N = atoi(argv[1]);
    unsigned int mem_size = N * sizeof(float);
    unsigned int blocksize = 256;

    // variables to calc runtimes
    unsigned long int elapsedP;
    struct timeval tStartP;
    struct timeval tEndP;
    struct timeval tDiffP;

    unsigned long int elapsedS;
    struct timeval tStartS;
    struct timeval tEndS;
    struct timeval tDiffS;

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

    // running the kernel
    gettimeofday(&tStartP, NULL);

    // execute kernel
    for(int i = 0; i < GPU_RUNS; i++) {
        squareKernel<<< (blocksize - 1 + N) / blocksize, blocksize >>>(d_in, gpu_out);
    }
    
    gettimeofday(&tEndP, NULL);
    timevalSubstract(&tDiffP, &tEndP, &tStartP);
    elapsedP = (tDiffP.tv_sec*1e6 + tDiffP.tv_usec) / GPU_RUNS;

    // running the sequential function
    gettimeofday(&tStartS, NULL);

    // execute the seq
    for(int i = 0; i < GPU_RUNS; i++) {
        squareSeq(N, cpu_out);
    }

    gettimeofday(&tEndS, NULL);
    timevalSubstract(&tDiffS, &tEndS, &tStartS);
    elapsedS = (tDiffS.tv_sec*1e6 + tDiffS.tv_usec) / GPU_RUNS;

    // copy result from device to host
    cudaMemcpy(h_out, gpu_out, mem_size, cudaMemcpyDeviceToHost);

    int test = 1;
    // print result
    for (unsigned int i = 0; i < N; ++i) {
        if (cpu_out[i] - h_out[i] > 0.0000001) {
            printf("surt kurt, der er sgu for stor forskel\n");
            printf("%-6f - %.6f\n", cpu_out[i], h_out[i]);
            test = 0;
        }
    }

    if(test) {
        printf("det virker sgu\n");
        printf("Parallel took  :  %d microseconds (%.2fms)\n", elapsedP, elapsedP / 1000.0);
        printf("Sequential took:  %d microseconds (%.2fms)\n", elapsedS, elapsedS / 1000.0);
    }

    // clean-up memory
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(gpu_out);
}