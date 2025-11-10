#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "kmeans.h"
#include "alloc.h"
#include "error.h"

// timing helper you already have
extern double wtime();

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) error("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
}
inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

// ----------------------------------------------------------------------------
// Row-major distance: objects [numObjs][numCoords], clusters [numClusters][numCoords]
// ----------------------------------------------------------------------------
__host__ __device__ inline static
double euclid_dist_2_rowmajor(int numCoords,
                              const double *objects,    // [numObjs][numCoords]
                              const double *clusters,   // [numClusters][numCoords]
                              int objectId,
                              int clusterId) {
  const double* obj = objects  + (size_t)objectId  * numCoords;
  const double* clu = clusters + (size_t)clusterId * numCoords;
  double ans = 0.0;
  for (int j = 0; j < numCoords; ++j) {
    double diff = obj[j] - clu[j];
    ans += diff * diff;
  }
  return ans;
}

__device__ inline int get_tid() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

// ----------------------------------------------------------------------------
// Kernel: assign to nearest cluster, accumulate sums and delta (row-major)
// deviceobjects:   [numObjs][numCoords]   row-major
// devicenewClusters: [numClusters][numCoords] row-major (sums)
// deviceClusters:  [numClusters][numCoords] row-major (centroids)
// ----------------------------------------------------------------------------
__global__ void find_nearest_cluster_rowmajor(int numCoords,
                                              int numObjs,
                                              int numClusters,
                                              const double *deviceobjects,     // [numObjs][numCoords]
                                              int *devicenewClusterSize,      // [numClusters]
                                              double *devicenewClusters,      // [numClusters][numCoords]
                                              const double *deviceClusters,    // [numClusters][numCoords]
                                              int *deviceMembership,          // [numObjs]
                                              double *devdelta) {
  extern __shared__ double shmem_total[];
  // shared layout: [clusters into shared] + [block reduction buffer]
  double *shmemClusters = shmem_total; // size: numClusters * numCoords
  double *delta_reduce  = shmem_total + (size_t)numClusters * numCoords;

  // load centroids into shared (row-major, contiguous)
  for (int i = threadIdx.x; i < numClusters * numCoords; i += blockDim.x) {
    shmemClusters[i] = deviceClusters[i];
  }
  __syncthreads();

  const int tid  = get_tid();
  const int lane = threadIdx.x;

  double mydelta = 0.0;

  if (tid < numObjs) {
    // find nearest cluster
    int best = 0;
    double best_dist = euclid_dist_2_rowmajor(numCoords, deviceobjects, shmemClusters, tid, 0);
    for (int c = 1; c < numClusters; ++c) {
      double d = euclid_dist_2_rowmajor(numCoords, deviceobjects, shmemClusters, tid, c);
      if (d < best_dist) { best_dist = d; best = c; }
    }

    // membership change?
    int old = deviceMembership[tid];
    if (old != best) mydelta = 1.0;
    deviceMembership[tid] = best;

    // accumulate this object's coords into the chosen cluster sums (row-major)
    atomicAdd(&devicenewClusterSize[best], 1);
    const double* obj = deviceobjects + (size_t)tid * numCoords;
    double* sumRow = devicenewClusters + (size_t)best * numCoords;
    for (int j = 0; j < numCoords; ++j) {
      atomicAdd(&sumRow[j], obj[j]);
    }
  }

  // block reduction for delta (ensure all threads participate)
  delta_reduce[lane] = mydelta;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (lane < s) delta_reduce[lane] += delta_reduce[lane + s];
    __syncthreads();
  }
  if (lane == 0) atomicAdd(devdelta, delta_reduce[0]);
}

// ----------------------------------------------------------------------------
// Kernel: divide sums by counts to form new centroids (row-major) and reset sums
// ----------------------------------------------------------------------------
__global__ void update_centroids_rowmajor(int numCoords,
                                          int numClusters,
                                          int *devicenewClusterSize,     // [numClusters]
                                          double *devicenewClusters,     // [numClusters][numCoords]
                                          double *deviceClusters)        // [numClusters][numCoords]
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = numClusters * numCoords;

  if (tid < total) {
    int clusterId = tid / numCoords; // row index
    if (devicenewClusterSize[clusterId] > 0) {
      deviceClusters[tid] = devicenewClusters[tid] / (double)devicenewClusterSize[clusterId];
    }
    // reset for next iteration
    devicenewClusters[tid] = 0.0;
  }
  __syncthreads();
  if (tid < numClusters) {
    devicenewClusterSize[tid] = 0;
  }
}

// ----------------------------------------------------------------------------
// Host driver: NO TRANSPOSE. Works directly with CPU row-major arrays.
// ----------------------------------------------------------------------------
void kmeans_gpu(double *objects,      /* in:  [numObjs][numCoords] (row-major) */
                int numCoords,
                int numObjs,
                int numClusters,
                double threshold,
                long loop_threshold,
                int *membership,      /* out: [numObjs] */
                double *clusters,     /* out: [numClusters][numCoords] (row-major) */
                int blockSize) {

  double timing = wtime(), timing_internal, timer_min = 1e42, timer_max = 0.0;
  double total_timing_gpu = 0.0;
  int loop = 0;

  printf("\n|-----------Full-offload GPU KMeans (row-major, no transpose)------------|\n\n");

  // Device buffers (row-major everywhere)
  double *deviceObjects      = nullptr;
  double *deviceClusters     = nullptr;
  double *devicenewClusters  = nullptr;
  int    *deviceMembership   = nullptr;
  int    *devicenewClusterSize = nullptr;
  double *dev_delta_ptr      = nullptr;


  deviceObjects = objects; //(typeof(objects)) malloc(numObjs * numCoords * sizeof(*objects));
  deviceClusters = clusters; //(double*)  malloc(numClusters * numCoords * sizeof(double));
  devicenewClusters = (double*)  malloc(numClusters * numCoords * sizeof(double));
  devicenewClusterSize = (int*)  malloc(numClusters * sizeof(int));
  deviceMembership = membership; //(int*) malloc(numObjs * sizeof(int));
  dev_delta_ptr = (double*) malloc(sizeof(double));

  for (int i = 0; i < numClusters * numCoords; i++) {
    devicenewClusters[i] = 0.0;
  }
  for (int i = 0; i < numClusters; i++) {
    devicenewClusterSize[i] = 0;
  }
  for (int i = 0; i < numObjs; i++) {
    deviceMembership[i] = -1;
  }

  // Launch configuration
  const unsigned int numThreadsPerBlock = (numObjs > blockSize) ? blockSize : numObjs;
  const unsigned int numBlocks = (numObjs + numThreadsPerBlock - 1) / numThreadsPerBlock;

  // Shared memory: centroids (numClusters*numCoords doubles) + per-thread delta buffer (blockDim doubles)
  const size_t clusterBlockSharedDataSize =
      (size_t)numClusters * numCoords * sizeof(double) +
      (size_t)numThreadsPerBlock * sizeof(double);

  // Hardware check
  cudaDeviceProp deviceProp;
  int deviceNum = 0;
  checkCuda(cudaGetDevice(&deviceNum));
  checkCuda(cudaGetDeviceProperties(&deviceProp, deviceNum));
  if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
    error("Insufficient shared memory per block for cluster centroids: need %zu bytes, have %zu bytes\n",
          clusterBlockSharedDataSize, (size_t)deviceProp.sharedMemPerBlock);
  }

  // Main loop
  do {
    timing_internal = wtime();

    checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(double)));

    double loop_gpu_time = wtime();
    find_nearest_cluster_rowmajor<<<numBlocks, numThreadsPerBlock, clusterBlockSharedDataSize>>>(
        numCoords, numObjs, numClusters,
        deviceObjects, devicenewClusterSize, devicenewClusters,
        deviceClusters, deviceMembership, dev_delta_ptr);
    checkCuda(cudaDeviceSynchronize());
    checkLastCudaError();
    loop_gpu_time = (wtime() - loop_gpu_time) * 1e3;

    // update centroids
    const unsigned int updBlock = ((unsigned int)(numClusters * numCoords) > (unsigned int)blockSize)
                                  ? (unsigned int)blockSize
                                  : (unsigned int)(numClusters * numCoords);
    const unsigned int updGrid  = ((unsigned int)(numClusters * numCoords) + updBlock - 1) / updBlock;

    double t2 = wtime();
    update_centroids_rowmajor<<<updGrid, updBlock>>>(
        numCoords, numClusters, devicenewClusterSize, devicenewClusters, deviceClusters);
    checkCuda(cudaDeviceSynchronize());
    checkLastCudaError();
    loop_gpu_time += (wtime() - t2) * 1e3;

    total_timing_gpu += loop_gpu_time;

    // bring back delta and check progress
    double delta = 0.0;
    checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(double), cudaMemcpyDeviceToHost));
    delta /= (double)numObjs;
    ++loop;

    timing_internal = 1e3 * (wtime() - timing_internal);
    if (timing_internal < timer_min) timer_min = timing_internal;
    if (timing_internal > timer_max) timer_max = timing_internal;

    if (delta <= threshold) break;
  } while (loop < loop_threshold);

  // Copy final centroids back (row-major) and (optionally) memberships
  checkCuda(cudaMemcpy(clusters, deviceClusters,
                       (size_t)numClusters * numCoords * sizeof(double),
                       cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(membership, deviceMembership,
                       (size_t)numObjs * sizeof(int),
                       cudaMemcpyDeviceToHost));

  //printf("cluster[0]=%f\n", clusters[0]);
  printf("nloops = %d  : total = %lf ms\n\t-> t_gpu = %lf ms\n",
         loop, 10e3 * (wtime() - timing), total_timing_gpu);
 
}

