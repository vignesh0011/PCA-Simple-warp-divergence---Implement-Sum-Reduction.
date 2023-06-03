# PCA-Simple-warp-divergence---Implement-Sum-Reduction.
Refer to the kernel reduceUnrolling8 and implement the kernel reduceUnrolling16, in which each thread handles 16 data blocks. Compare kernel performance with reduceUnrolling8 and use the proper metrics and events with nvprof to explain any difference in performance.

## Aim:
To implement the kernel reduceUnrolling16 and comapare the performance of kernal reduceUnrolling16 with kernal reduceUnrolling8 using proper metrics and events with nvprof.
## Procedure:
Step 1 : Include the required files and library.

Step 2 : Introduce a function named 'recursiveReduce' to implement Interleaved Pair Approach and function 'reduceInterleaved' to implement Interleaved Pair with less divergence.

Step 3 : Introduce a function named 'reduceNeighbored' to implement Neighbored Pair with divergence and function 'reduceNeighboredLess' to implement Neighbored Pair with less divergence.

Step 4 : Introduce optimizations such as unrolling to reduce divergence.

Step 5 : Declare three global function named 'reduceUnrolling2' , 'reduceUnrolling4' , 'reduceUnrolling8' , 'reduceUnrolling16' and then set the thread ID , convert global data pointer to the local pointer of the block , perform in-place reduction in global memory ,finally write the result of the block to global memory in all the three function respectively.

Step 6 : Declare functions to unroll the warp. Declare a global function named 'reduceUnrollWarps8' and then set the thread ID , convert global data pointer to the local pointer of the block , perform in-place reduction in global memory , unroll the warp ,finally write the result of the block to global memory infunction.

Step 7 : Declare Main method/function . In the Main method , set up the device and initialise the size and block size. Allocate the host memory and device memory and then call the kernals decalred in the function.

Step 8 : Atlast , free the host and device memory then reset the device and check for results.

# Program:
kernel reduceUnrolling8
``` c
#include <stdio.h>
#include <cuda.h>

#define N 1024

__global__ void sumReduction(int* data) {
    __shared__ int sharedData[N];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = data[i];
    __syncthreads();

    // Perform reduce unrolling8 sum reduction using warp divergence
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Warp divergence
    if (tid < 32) {
        volatile int* vsharedData = sharedData;
        vsharedData[tid] += vsharedData[tid + 32];
        vsharedData[tid] += vsharedData[tid + 16];
        vsharedData[tid] += vsharedData[tid + 8];
        vsharedData[tid] += vsharedData[tid + 4];
        vsharedData[tid] += vsharedData[tid + 2];
        vsharedData[tid] += vsharedData[tid + 1];
    }

    // Store the final result in global memory
    if (tid == 0) {
        data[blockIdx.x] = sharedData[0];
    }
}

int main() {
    int* data;
    int* d_data;

    // Allocate and initialize the data array
    data = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        data[i] = i + 1;
    }

    // Allocate device memory for the data array
    cudaMalloc(&d_data, N * sizeof(int));

    // Copy the data array from host to device
    cudaMemcpy(d_data, data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 block(N / 2);
    dim3 grid(1);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    sumReduction << <grid, block >> > (d_data);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Copy the result back from device to host
    cudaMemcpy(data, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the final sum
    printf("Sum: %d\n", data[0]);

    // Calculate the elapsed time
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end);
    printf("Elapsed Time: %.6f milliseconds\n", elapsed);

    // Destroy the CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // Free device memory
    cudaFree(d_data);

    // Free host memory
    free(data);

    return 0;
}

```

unrolling16

``` c
#include <stdio.h>
#include <cuda.h>
#define N 1024
__global__ void sumReduction(int* data) {
    __shared__ int sharedData[N];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    // Load data into shared memory
    sharedData[tid] = data[i];
    __syncthreads();

    // Perform reduce unrolling16 sum reduction using warp divergence
    if (tid < 512) sharedData[tid] += sharedData[tid + 512];
    __syncthreads();
    if (tid < 256) sharedData[tid] += sharedData[tid + 256];
    __syncthreads();
    if (tid < 128) sharedData[tid] += sharedData[tid + 128];
    __syncthreads();
    if (tid < 64) sharedData[tid] += sharedData[tid + 64];
    __syncthreads();

    // Warp divergence
    if (tid < 32) {
        volatile int* vsharedData = sharedData;
        vsharedData[tid] += vsharedData[tid + 32];
        vsharedData[tid] += vsharedData[tid + 16];
        vsharedData[tid] += vsharedData[tid + 8];
        vsharedData[tid] += vsharedData[tid + 4];
        vsharedData[tid] += vsharedData[tid + 2];
        vsharedData[tid] += vsharedData[tid + 1];
    }

    // Store the final result in global memory
    if (tid == 0) {
        data[blockIdx.x] = sharedData[0];
    }
}

int main() {
    int* data;
    int* d_data;
    cudaEvent_t start, stop;
    float elapsed;

    // Allocate and initialize the data array
    data = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        data[i] = i + 1;
    }

    // Allocate device memory for the data array
    cudaMalloc(&d_data, N * sizeof(int));

    // Copy the data array from host to device
    cudaMemcpy(d_data, data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 block(N / 32);
    dim3 grid(1);

    // Start the timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    sumReduction << <grid, block >> > (d_data);

    // Stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    // Copy the result back from device to host
    cudaMemcpy(data, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the final sum
    printf("Sum: %d\n", data[0]);

    // Print the elapsed time
    printf("Elapsed Time: %.3f ms\n", elapsed);

    // Free device memory
    cudaFree(d_data);

    // Free host memory
    free(data);

    return 0;
}
```
## Output:
### Unrolling 8
![image](https://github.com/curiouzs/PCA-Simple-warp-divergence---Implement-Sum-Reduction./assets/75234646/ed14412b-a3fc-4f07-9530-9981c5637940)

### Unrolling 16
![image](https://github.com/curiouzs/PCA-Simple-warp-divergence---Implement-Sum-Reduction./assets/75234646/c2560952-ad1d-4860-ba98-de649b38af08)


## Result:
Implementation of the kernel reduceUnrolling16 is done and the performance of kernal reduceUnrolling16 is comapared with kernal reduceUnrolling8 using proper metrics and events with nvprof.
