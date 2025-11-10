#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"
#include "alloc.h"
#include "error.h"

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        error("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

__device__ int get_tid() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

/* square of Euclid distance between two multi-dimensional points using column-base format */
__host__ __device__ inline static
double euclid_dist_2_transpose(int numCoords,
                               int numObjs,
                               int numClusters,
                               double *objects,     // [numCoords][numObjs]
                               double *clusters,    // [numCoords][numClusters]
                               int objectId,
                               int clusterId) {
  int i;
  double ans = 0.0;

  for(i=0; i<numCoords; i++) {
        double diff = objects[numObjs*i + objectId] - clusters[clusterId + numClusters*i];
        ans += diff * diff;
  }
  return (ans);
}

__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          double *deviceobjects,           //  [numCoords][numObjs]
                          int *devicenewClusterSize,      //  [numClusters]
                          double *devicenewClusters,      //  [numCoords][numClusters]
                          double *deviceClusters,         //  [numCoords][numClusters]
                          int *deviceMembership,          //  [numObjs]
                          double *devdelta) {
  extern __shared__ double shmem_total[];
  double *shmemClusters = shmem_total;
  double *delta_reduce_buff = shmem_total + numClusters * numCoords;

  int tid1 = threadIdx.x;
    
  for (int i = tid1; i < numCoords * numClusters; i += blockDim.x) {
    shmemClusters[i] = deviceClusters[i];
  }
  __syncthreads();

  int tid = get_tid();

  if (tid < numObjs) {
    int index = 0;
    double min_dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, deviceobjects, shmemClusters, tid, 0);
    for (int i = 1; i < numClusters; i++) {
      double dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, deviceobjects, shmemClusters, tid, i);
      if (dist < min_dist) {
        min_dist = dist;
        index = i;
      }
    }

    delta_reduce_buff[threadIdx.x] = (deviceMembership[tid] != index) ? 1.0 : 0.0;  
    
    deviceMembership[tid] = index;

    atomicAdd(&devicenewClusterSize[index], 1);
    for (int j = 0; j < numCoords; j++) {
      atomicAdd(&devicenewClusters[j * numClusters + index], deviceobjects[j * numObjs + tid]);
    }
  
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
      if (tid1 < i) delta_reduce_buff[tid1] += delta_reduce_buff[tid1 + i];
      __syncthreads();
      i /= 2;
    }

    if (tid1 == 0) atomicAdd(devdelta, delta_reduce_buff[0]);
  }
}

__global__ static
void update_centroids(int numCoords,
                      int numClusters,
                      int *devicenewClusterSize,           //  [numClusters]
                      double *devicenewClusters,    //  [numCoords][numClusters]
                      double *deviceClusters)    //  [numCoords][numClusters])
{
  int tid = get_tid();
  int clustersize_id = tid % numClusters;
  if(tid < numClusters*numCoords) {
    if(devicenewClusterSize[clustersize_id] > 0) { 
      deviceClusters[tid] = devicenewClusters[tid]/devicenewClusterSize[clustersize_id];
    }
    devicenewClusters[tid] = 0; // reset for next iteration
  }
  __syncthreads();
  if (tid < numClusters) {
    devicenewClusterSize[tid] = 0;
  }
}


void kmeans_gpu(double *objects,      /* in: [numObjs][numCoords] */
                int numCoords,
                int numObjs,
                int numClusters,
                double threshold,
                long loop_threshold,
                int *membership,      /* out: [numObjs] */
                double *clusters,     /* out: [numClusters][numCoords] */
                int blockSize) {

  double timing = wtime(), timing_internal, timer_min = 1e42, timer_max = 0;
  double cpu_time = 0, timing_cpu = 0, timing_gpu = 0, delta = 0, *dev_delta_ptr;
  int loop = 0;

  printf("\n|-----------Full-offload GPU KMeans (managed memory)------------|\n\n");
  
  double *deviceObjects;
  double *deviceClusters, *devicenewClusters;
  int *deviceMembership;
  int *devicenewClusterSize;

  double t_alloc = wtime();
  checkCuda(cudaMallocManaged(&deviceObjects, numObjs * numCoords * sizeof(double)));
  checkCuda(cudaMallocManaged(&deviceClusters, numClusters * numCoords * sizeof(double)));
  checkCuda(cudaMallocManaged(&devicenewClusters, numClusters * numCoords * sizeof(double)));
  checkCuda(cudaMallocManaged(&devicenewClusterSize, numClusters * sizeof(int)));
  checkCuda(cudaMallocManaged(&deviceMembership, numObjs * sizeof(int)));
  checkCuda(cudaMallocManaged(&dev_delta_ptr, sizeof(double)));
  
  t_alloc = wtime() - t_alloc;

  double t_init = wtime();
  //column-major
  for (int i = 0; i < numObjs; i++) {
    for (int j = 0; j < numCoords; j++) {
      deviceObjects[j * numObjs + i] = objects[i * numCoords + j];
    }
  }

  //initialize cluster centers from first 'numClusters' objects
  for (int i = 0; i < numClusters; i++) {
    for (int j = 0; j < numCoords; j++) {
      deviceClusters[j * numClusters + i] = deviceObjects[j * numObjs + i];
    }
  }

  for (int i = 0; i < numClusters * numCoords; i++) {
    devicenewClusters[i] = 0.0;
  }
  for (int i = 0; i < numClusters; i++) {
    devicenewClusterSize[i] = 0;
  }
  for (int i = 0; i < numObjs; i++) {
    deviceMembership[i] = -1;
  }

  t_init = wtime()-t_init;

  double t1 = wtime();
  const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize) ? blockSize : numObjs;
  const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
  const unsigned int clusterBlockSharedDataSize =
      (numClusters * numCoords * sizeof(double) + numThreadsPerClusterBlock * sizeof(double));

  // Check shared memory sufficiency
  cudaDeviceProp deviceProp;
  int deviceNum;
  cudaGetDevice(&deviceNum);
  cudaGetDeviceProperties(&deviceProp, deviceNum);

  if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
    error("Your CUDA hardware has insufficient block shared memory to hold all cluster centroids\n");
  }

  t1 = wtime() - t1;
  cpu_time += t1*1e3;

double total_timing_gpu = 0.0;

do {
  timing_internal = wtime();
  double t2 = wtime();
  checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(double)));
  t2 = 1e3*(wtime() - t2);

  double loop_gpu_time = wtime();
  find_nearest_cluster<<<numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize>>>(
      numCoords, numObjs, numClusters,
      deviceObjects, devicenewClusterSize, devicenewClusters,
      deviceClusters, deviceMembership, dev_delta_ptr);
  checkCuda(cudaDeviceSynchronize());
  checkLastCudaError();
  loop_gpu_time = (wtime() - loop_gpu_time)*1e3;

  t1 = wtime();
  delta = *dev_delta_ptr;

  const unsigned int update_centroids_block_sz = (numCoords * numClusters > blockSize) ? blockSize : numCoords * numClusters;
  const unsigned int update_centroids_dim_sz = (numCoords * numClusters + update_centroids_block_sz - 1) / update_centroids_block_sz;
  t1 = wtime() - t1;
  cpu_time += t1*1e3;
  double t_gpu2 = wtime();
  update_centroids<<<update_centroids_dim_sz, update_centroids_block_sz>>>(
      numCoords, numClusters, devicenewClusterSize, devicenewClusters, deviceClusters);
  checkCuda(cudaDeviceSynchronize());
  checkLastCudaError();
  loop_gpu_time += (wtime() - t_gpu2)*1e3;

  total_timing_gpu += loop_gpu_time + t2;

    timing_cpu = wtime();
    delta /= numObjs;
    //printf("delta is %f - ", delta);
    loop++;
    //printf("completed loop %d\n", loop);
    cpu_time += wtime() - timing_cpu;


  timing_internal = 1e3 * (wtime() - timing_internal);
  if (timing_internal < timer_min) timer_min = timing_internal;
  if (timing_internal > timer_max) timer_max = timing_internal;

} while (delta > threshold && loop < loop_threshold);

  timing_cpu=wtime();
  for (int i = 0; i < numClusters; i++) {
    for (int j = 0; j < numCoords; j++) {
      clusters[i * numCoords + j] = deviceClusters[j * numClusters + i];
    }
  }
  cpu_time += 1e3*(wtime() - timing_cpu);


  printf("nloops = %d  : end2end = %lf ms\n\t-> t_alloc = %lf ms\n\t-> t_init = %lf ms\n\t-> t_cpu = %lf ms\n\t-> t_gpu = %lf ms\n\t",
         loop, 1e3*(wtime() - timing), 1e3*t_alloc, 1e3*t_init, cpu_time, total_timing_gpu);

  cudaFree(deviceObjects);
  cudaFree(deviceClusters);
  cudaFree(devicenewClusters);
  cudaFree(devicenewClusterSize);
  cudaFree(deviceMembership);
  cudaFree(dev_delta_ptr);
}


