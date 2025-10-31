//modes: explicit, explicit_async, um_migrate, gh_hbm_shared, gh_cpu_shared, gh_hmm_pageable

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <unistd.h>   
#include <sys/time.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)
#endif

double wtime(void) 
{
    double now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +              // in seconds
               ((double)etstart.tv_usec) / 1000000.0;  // in microseconds
    return now_time;
}

//  ****************FOR HMM_PAGEABLE (SYSTEM ALLOCATED) MODE: GPU INITS THE RANDOM MATRIX ON SYSTEM MEMORY AFTER THE MATRIXES IS MALLOC()-ED BY CPU ************************************

// 64-bit SplitMix: small, fast, repeatable per index
__device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Convert 64-bit to uniform [0,1) as double using top 53 bits
__device__ inline double u01_from_u64(uint64_t x) {
  const double inv2_53 = 1.0 / 9007199254740992.0; // 2^53
  return double(x >> 11) * inv2_53;
}

// Fill array with random doubles in [-1, 1]
__global__ void init_random_double(double* a, size_t n, uint64_t seed) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) {
    uint64_t r = splitmix64(seed + i);
    double u = u01_from_u64(r);   // [0,1)
    a[i] = 2.0 * u - 1.0;         // [-1,1)
  }
}

// Set array to zero
__global__ void set_zero_double(double* a, size_t n) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) a[i] = 0.0;
}

//  ********************************************************************************************************************************************************************************

//bank-conflict-free tiled transpose
template <typename T, int TILE=32, int BLOCK_ROWS=8>
__global__ void transpose_tiled(const T* __restrict__ A, T* __restrict__ B, int N) {
  __shared__ T tile[TILE][TILE+1]; // +1 to avoid bank conflicts on column reads

  //for each thread
  int x = blockIdx.x * TILE + threadIdx.x; // column in A
  int y = blockIdx.y * TILE + threadIdx.y; // row in A

  //load A -> shared tile (coalesced)
  #pragma unroll
  for (int i = 0; i < TILE; i += BLOCK_ROWS) {
    int yy = y + i;
    if (x < N && yy < N)
      tile[threadIdx.y + i][threadIdx.x] = A[yy * (size_t)N + x]; //during the same loop iteration, all threads have the same yy and consecutive x's(row major)
  }
  __syncthreads();

  //write (shared)^T -> B (coalesced)
  //first we find the transposed position of each tile inside the matrix B
  int xt = blockIdx.y * TILE + threadIdx.x; // column in B
  int yt = blockIdx.x * TILE + threadIdx.y; // row in B
  //then, we compute the new transposed posistion of each element inside the tile
  #pragma unroll
  for (int i = 0; i < TILE; i += BLOCK_ROWS) {
    int yy = yt + i;
    if (xt < N && yy < N)
      B[yy * (size_t)N + xt] = tile[threadIdx.x][threadIdx.y + i];
  }
}


enum class Mode {
  EXPLICIT,        //device malloc + memcpy on pinned host
  EXPLICIT_ASYNC,  //same as above but with cuda malloc async
  UM_MIGRATE,      //managed + prefetch to GPU; prefetch result back to CPU
  GH_HBM_SHARED,   //managed preferred on GPU; CPU reads GPU HBM coherently (no post-prefetch)
  GH_CPU_SHARED,   //pinned host + device pointer (zero-copy: GPU accesses host)
  GH_HMM_PAGEABLE,  //plain malloc() pageable host; GPU accesses via HMM/page faulting
  GH_HMM_PAGEABLE_CUDA_INIT, //same as above, but now the matrix gets initialized by the gpu on system memory
  GH_HBM_SHARED_NO_PREFETCH,
  UM_MIGRATE_NO_PREFETCH
};

static Mode parse_mode_str(const std::string& s) {
  if (s == "explicit")        return Mode::EXPLICIT;
  if (s == "explicit_async")        return Mode::EXPLICIT_ASYNC;
  if (s == "um_migrate")      return Mode::UM_MIGRATE;
  if (s == "gh_hbm_shared")   return Mode::GH_HBM_SHARED;
  if (s == "gh_cpu_shared")   return Mode::GH_CPU_SHARED;
  if (s == "gh_hmm_pageable") return Mode::GH_HMM_PAGEABLE;
  if (s == "gh_hmm_pageable_cuda_init") return Mode::GH_HMM_PAGEABLE_CUDA_INIT;
  if (s == "gh_hbm_shared_no_prefetch") return Mode::GH_HBM_SHARED_NO_PREFETCH;
  if (s == "um_migrate_no_prefetch") return Mode::UM_MIGRATE_NO_PREFETCH;
  fprintf(stderr, "Unknown --mode=%s\n", s.c_str());
  std::exit(EXIT_FAILURE);
}


struct Args {
  int iters = 10;
  int N = 4096;
  std::string mode = "explicit";
  int prefetch = 1;     // only for UM_MIGRATE or GH_HBM_SHARED
  uint64_t seed = 12345;
};

static void usage(const char* prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  -i INT    iterations (default 10)\n"
    "  -n INT    matrix order N (default 4096)\n"
    "  -m STR    mode: explicit | um_migrate | gh_hbm_shared | gh_cpu_shared | gh_hmm_pageable (default explicit)\n"
    "  -p 0|1    prefetch for UM/GH_HBM_SHARED (default 1)\n"
    "  -r INT    RNG seed (default 12345)\n"
    "  -h        help\n", prog);
}

static Args parse(int argc, char** argv) {
  Args a;
  int opt;
  while ((opt = getopt(argc, argv, "i:n:m:p:r:h")) != -1) {
    switch (opt) {
      case 'i': a.iters = std::atoi(optarg); break;
      case 'n': a.N     = std::atoi(optarg); break;
      case 'm': a.mode  = optarg; break;
      case 'p': a.prefetch = std::atoi(optarg); break;
      case 'r': a.seed  = (uint64_t)std::strtoull(optarg, nullptr, 10); break;
      case 'h': usage(argv[0]); std::exit(EXIT_SUCCESS);
      default:  usage(argv[0]); std::exit(EXIT_FAILURE);
    }
  }
  if (a.iters <= 0 || a.N <= 0) {
    fprintf(stderr, "Error: iterations and N must be > 0\n");
    std::exit(EXIT_FAILURE);
  }
  return a;
}

//validation helpers
static double max_abs_diff_transpose(const double* A, const double* B, int N) {
  // max |B[i,j] - A[j,i]|
  double m = 0.0;
  for (int i = 0; i < N; ++i) {
    size_t baseB = (size_t)i * N;
    for (int j = 0; j < N; ++j) {
      double diff = std::fabs(B[baseB + j] - A[(size_t)j * N + i]);
      if (diff > m) m = diff;
    }
  }
  return m;
}

template <typename T>
static double checksum(const T* p, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) s += (double)p[i];
  return s;
}


int main(int argc, char** argv) {
  Args args = parse(argc, argv);
  Mode mode = parse_mode_str(args.mode);

  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));  
  
  
  //HMM capability queried for pageable mode
  int pageable = 0, uses_host_pt = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev));
  CHECK_CUDA(cudaDeviceGetAttribute(&uses_host_pt, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev));

  const int N = args.N;
  const size_t elems = (size_t)N * N;
  const size_t bytes = elems * sizeof(double);

  printf("iters=%d N=%d (%.3f MiB) mode=%s prefetch=%d seed=%llu\n",
         args.iters, N, double(bytes)/(1024.0*1024.0),
         args.mode.c_str(), args.prefetch, (unsigned long long)args.seed);

  //Host and device-visible pointers
  double *hA = nullptr, *hB = nullptr;
  double *dA = nullptr, *dB = nullptr;
  cudaStream_t stream = 0;

  //allocation by mode
  switch (mode) {
    case Mode::EXPLICIT: { //allocate pinned(not pageable) memory on host
      CHECK_CUDA(cudaMallocHost(&hA, bytes));
      CHECK_CUDA(cudaMallocHost(&hB, bytes));
      CHECK_CUDA(cudaMalloc(&dA, bytes));
      CHECK_CUDA(cudaMalloc(&dB, bytes));
      break;
			 }
    case Mode::EXPLICIT_ASYNC: {
      CHECK_CUDA(cudaMallocAsync((void**)&dA, bytes, stream));
      CHECK_CUDA(cudaMallocAsync((void**)&dB, bytes, stream));
      CHECK_CUDA(cudaMallocHost(&hA, bytes));
      CHECK_CUDA(cudaMallocHost(&hB, bytes));
      break;
			       }
    case Mode::UM_MIGRATE: // the same allocation style with GH_HBM_SHARED(managed memory)
    case Mode::UM_MIGRATE_NO_PREFETCH: {
      CHECK_CUDA(cudaMallocManaged(&dA, bytes));
      CHECK_CUDA(cudaMallocManaged(&dB, bytes));
      hA = dA; hB = dB; //same pointer for CPU and GPU
      break;
				       }
    case Mode::GH_HBM_SHARED_NO_PREFETCH:  
    case Mode::GH_HBM_SHARED: {
      CHECK_CUDA(cudaMallocManaged(&dA, bytes));
      CHECK_CUDA(cudaMallocManaged(&dB, bytes));
      hA = dA; hB = dB; //same pointer for CPU and GPU

      cudaMemLocation loc_dev{}; loc_dev.type = cudaMemLocationTypeDevice; loc_dev.id = dev;

      CHECK_CUDA(cudaMemAdvise(dA, bytes, cudaMemAdviseSetPreferredLocation, loc_dev)); //hint to UM pager:If you have a choice, keep these pages resident on GPU dev,it’s a policy hint, 
                                                                                    //not a hard pin. If the CPU (or another GPU) starts accessing those pages a lot, the runtime may 
                                                                                    //still migrate or replicate them. But with this hint, after GPU use, the pager will try to keep or return the pages 
                                                                                    //to GPU memory (HBM) rather than drifting back to system RAM.
      CHECK_CUDA(cudaMemAdvise(dB, bytes, cudaMemAdviseSetPreferredLocation, loc_dev));
      cudaMemLocation loc_cpu{}; loc_cpu.type = cudaMemLocationTypeHost; loc_cpu.id = 0;

      CHECK_CUDA(cudaMemAdvise(dA, bytes, cudaMemAdviseSetAccessedBy, loc_cpu)); //This can reduce first-touch penalties when the CPU later reads managed memory.
      CHECK_CUDA(cudaMemAdvise(dB, bytes, cudaMemAdviseSetAccessedBy, loc_cpu)); //  --//--

      break;
			      }
    case Mode::GH_CPU_SHARED:  {     //GPU will access host memory directly over PCIe/NVLink (zero-copy), this will be probably one of the slower ones due to limited bandwidth
      CHECK_CUDA(cudaMallocHost(&hA, bytes));
      CHECK_CUDA(cudaMallocHost(&hB, bytes));
      CHECK_CUDA(cudaHostGetDevicePointer(&dA, hA, 0)); 
      CHECK_CUDA(cudaHostGetDevicePointer(&dB, hB, 0));
      break;
			       }
    case Mode::GH_HMM_PAGEABLE_CUDA_INIT: //same alocated with malloc by the host, the initialization differs
    case Mode::GH_HMM_PAGEABLE: { //plain malloc(I think this falls into the category of system allocated memory), the GPU touches CPU pages on demand(first touch overhead maybe)
      if (!pageable) {
        fprintf(stderr, "ERROR: Device lacks cudaDevAttrPageableMemoryAccess; HMM pageable not supported.\n");
        return 1;
      }
      hA = (double*)std::malloc(bytes);
      hB = (double*)std::malloc(bytes);
      if (!hA || !hB) { fprintf(stderr, "malloc failed\n"); return 1; }
      dA = hA; dB = hB; //same raw pointer, GPU will fault pages from CPU on demand
      printf("HMM pageable supported (uses_host_page_tables=%d)\n", uses_host_pt);
      break;
				}
  }

  //random init on cpu
  if(mode != Mode::GH_HMM_PAGEABLE_CUDA_INIT)
  {
    std::mt19937_64 rng(args.seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < elems; ++i) hA[i] = dist(rng);
    std::fill(hB, hB + elems, 0.0);
  }
  else {
	// GPU init into pageable host memory via HMM
  	const int threads = 256;
  	const int blocksA = int((elems + threads - 1) / threads);
  	const int blocksB = blocksA;

  	init_random_double<<<blocksA, threads>>>(dA, elems, args.seed);
  	set_zero_double<<<blocksB, threads>>>(dB, elems);
  	CHECK_CUDA(cudaGetLastError());
  	CHECK_CUDA(cudaDeviceSynchronize());
  
  }

  //prefetch / copies before kernel
  if (mode == Mode::UM_MIGRATE || mode == Mode::GH_HBM_SHARED) {
    if (args.prefetch) {
      cudaMemLocation dst_dev{}; dst_dev.type = cudaMemLocationTypeDevice; dst_dev.id = dev;
      CHECK_CUDA(cudaMemPrefetchAsync(dA, bytes, dst_dev, 0));
      CHECK_CUDA(cudaMemPrefetchAsync(dB, bytes, dst_dev, 0));
      CHECK_CUDA(cudaDeviceSynchronize());
    }
  } else if (mode == Mode::EXPLICIT || mode == Mode::EXPLICIT_ASYNC) {
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dB, 0, bytes));
  }
  else if (mode == Mode::UM_MIGRATE_NO_PREFETCH || mode == Mode::GH_HBM_SHARED_NO_PREFETCH) {
    // nothing: managed pages will fault/migrate on first touch
  }	

  constexpr int TILE = 32;
  constexpr int BLOCK_ROWS = 8;
  dim3 block(TILE, BLOCK_ROWS);
  dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);


  /*
  //warm-up
  
  transpose_tiled<double, TILE, BLOCK_ROWS><<<grid, block>>>(dA, dB, N);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  */


  //timed loop
  std::vector<float> times(args.iters);
  for (int it = 0; it < args.iters; ++it) {
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    transpose_tiled<double, TILE, BLOCK_ROWS><<<grid, block>>>(dA, dB, N);
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
    times[it] = ms;
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
  }
  double avg_ms = 0.0;
  for (float t : times) avg_ms += t;
  avg_ms /= times.size();
  const double bytes_moved = 2.0 * double(bytes); // read + write
  const double gbps = bytes_moved / (1.0e6 * avg_ms);
  printf("Kernel avg: %.3f ms, effective bandwidth: %.2f GB/s\n", avg_ms, gbps);

  //UM migrate: prefetch back to CPU to time migration
  if (mode == Mode::UM_MIGRATE) {
    cudaMemLocation dst_cpu{}; dst_cpu.type = cudaMemLocationTypeHost; dst_cpu.id = 0;
    auto t0 = wtime();
    CHECK_CUDA(cudaMemPrefetchAsync(dB, bytes, dst_cpu, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t1 = wtime();
    double ms = (t1 - t0)*1e3;
    printf("UM prefetch-to-CPU: %.3f ms (%.2f GB/s logical)\n", ms, bytes / 1e6 / ms);
  }

  
  auto r0 = wtime();
  double sumB = checksum(hB, elems);
  auto r1 = wtime();
  double read_ms = (r1 - r0) * 1e3;
  printf("CPU checksum(B)=%.6e, CPU read time: %.3f ms (%.2f GB/s logical)\n",
         sumB, read_ms, bytes / 1e6 / read_ms);

  if (mode == Mode::EXPLICIT || mode == Mode::EXPLICIT_ASYNC) {
    double t0 = wtime();
    CHECK_CUDA(cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost));
    double t1 = wtime();
    double ms = (t1 - t0) * 1e3;
    printf("D2H memcpy: %.3f ms (%.2f GB/s)\n", ms, bytes / 1e6 / ms);
  }

  //correctness check: max |B - A^T|
  double maxerr = max_abs_diff_transpose(hA, hB, N);
  printf("Max |B - A^T| = %.3e  => %s\n", maxerr, (maxerr < 1e-9 ? "OK" : "MISMATCH"));


  switch (mode) {
    case Mode::EXPLICIT:
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFreeHost(hA));
      CHECK_CUDA(cudaFreeHost(hB));
      break;
    case Mode::EXPLICIT_ASYNC:
      CHECK_CUDA(cudaFreeAsync(dA, stream));
      CHECK_CUDA(cudaFreeAsync(dB, stream));
      CHECK_CUDA(cudaFreeHost(hA));
      CHECK_CUDA(cudaFreeHost(hB));
      break;
    case Mode::UM_MIGRATE:
    case Mode::GH_HBM_SHARED_NO_PREFETCH:
    case Mode::UM_MIGRATE_NO_PREFETCH:
    case Mode::GH_HBM_SHARED:
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      break;
    case Mode::GH_CPU_SHARED:
      CHECK_CUDA(cudaFreeHost(hA));
      CHECK_CUDA(cudaFreeHost(hB));
      break;
    case Mode::GH_HMM_PAGEABLE_CUDA_INIT:
    case Mode::GH_HMM_PAGEABLE:
      std::free(hA);
      std::free(hB);
      break;
  }

  return 0;
}

