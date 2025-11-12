//transpose_hybrid_benchmark.cu
//hybrid CPU+GPU tiled transpose with 7 memory modes (explicit, UM variants, GH_*).
//CPU handles columns [0:k), GPU handles columns [k:N). Both run concurrently.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <thread>
#include <atomic>
#include <unistd.h>     
#include <sys/time.h>   
#include <omp.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)
#endif

static inline double wtime(void)
{
  struct timeval  tv; struct timezone tz{};
  if (gettimeofday(&tv, &tz) == -1) perror("gettimeofday failed");
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

//validation 
static double max_abs_diff_transpose(const double* A, const double* B, int N) {
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


__device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
__device__ inline double u01_from_u64(uint64_t x) {
  const double inv2_53 = 1.0 / 9007199254740992.0; // 2^53
  return double(x >> 11) * inv2_53;
}
__global__ void init_random_double(double* a, size_t n, uint64_t seed) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) {
    uint64_t r = splitmix64(seed + i);
    double u = u01_from_u64(r);
    a[i] = 2.0 * u - 1.0; // [-1,1)
  }
}
__global__ void set_zero_double(double* a, size_t n) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) a[i] = 0.0;
}


template <typename T, int TILE=32, int BLOCK_ROWS=8>
__global__ void transpose_tiled_rect(const T* __restrict__ A, T* __restrict__ B,
                                     int N, int x0, int y0, int W, int H)
{
  __shared__ T tile[TILE][TILE+1];

  int block_x = blockIdx.x * TILE;
  int block_y = blockIdx.y * TILE;
  int tx = threadIdx.x, ty = threadIdx.y;

  int validW = max(0, min(TILE, W - block_x));
  int validH = max(0, min(TILE, H - block_y));

  int ax0 = x0 + block_x + tx;
  int ay0 = y0 + block_y + ty;

  #pragma unroll
  for (int i = 0; i < TILE; i += BLOCK_ROWS) {
    if (tx < validW && ty + i < validH)
      tile[ty + i][tx] = A[(size_t)(ay0 + i) * N + ax0];
  }
  __syncthreads();

  int bx  = y0 + block_y + tx;     // column in B
  int by0 = x0 + block_x + ty;     // base row in B

  #pragma unroll
  for (int i = 0; i < TILE; i += BLOCK_ROWS) {
    if (tx < validH && ty + i < validW)
      B[(size_t)(by0 + i) * N + bx] = tile[tx][ty + i];
  }
}

static void transpose_cpu_blocked(const double* __restrict__ A,
                                  double* __restrict__ B,
                                  int N, int c0, int c1, int BS=32)
{
  for (int jj = c0; jj < c1; jj += BS) {
    int Jmax = std::min(jj + BS, c1);
    for (int ii = 0; ii < N; ii += BS) {
      int Imax = std::min(ii + BS, N);
      for (int j = jj; j < Jmax; ++j) {
        const size_t jN = (size_t)j * N;
        for (int i = ii; i < Imax; ++i) {
          B[jN + i] = A[(size_t)i * N + j];
        }
      }
    }
  }
}


enum class Mode {
  EXPLICIT,
  UM_MIGRATE,
  GH_HBM_SHARED,
  GH_CPU_SHARED,
  GH_HMM_PAGEABLE,
  GH_HMM_PAGEABLE_CUDA_INIT,
  UM_MIGRATE_NO_PREFETCH,
  GH_HBM_SHARED_NO_PREFETCH
};

static Mode parse_mode_str(const std::string& s) {
  if (s == "explicit")                  return Mode::EXPLICIT;
  if (s == "um_migrate")                return Mode::UM_MIGRATE;
  if (s == "gh_hbm_shared")             return Mode::GH_HBM_SHARED;
  if (s == "gh_cpu_shared")             return Mode::GH_CPU_SHARED;
  if (s == "gh_hmm_pageable")           return Mode::GH_HMM_PAGEABLE;
  if (s == "gh_hmm_pageable_cuda_init")  return Mode::GH_HMM_PAGEABLE_CUDA_INIT;
  if (s == "um_migrate_no_prefetch")    return Mode::UM_MIGRATE_NO_PREFETCH;
  if (s == "gh_hbm_shared_no_prefetch") return Mode::GH_HBM_SHARED_NO_PREFETCH;
  fprintf(stderr, "Unknown --mode=%s\n", s.c_str()); std::exit(EXIT_FAILURE);
}

struct Args {
  int N = 8192;
  int iters = 10;         
  std::string mode = "explicit";
  int prefetch = 1;       
  uint64_t seed = 12345;
  double frac = 0.5; //this is how much the CPU takes, the GPU taked 1-frac
  int threads = 72;
};

static void usage(const char* prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  -n INT     matrix order N (default 8192)\n"
    "  -i INT     iterations (default 10)\n"
    "  -m STR     mode: explicit | um_migrate | gh_hbm_shared | gh_cpu_shared | gh_hmm_pageable |\n"
    "             um_migrate_no_prefetch | gh_hbm_shared_no_prefetch (default explicit)\n"
    "  -p 0|1     prefetch for UM/GH_HBM_SHARED (default 1)\n"
    "  -f FLOAT   GPU fraction in [0,1] (default 0.5) — GPU handles columns [f*N, N)\n"
    "  -r UINT64  RNG seed (default 12345)\n"
    "  -h         help\n", prog);
}

static Args parse(int argc, char** argv) {
  Args a; int opt;
  while ((opt = getopt(argc, argv, "n:i:m:p:f:r:h")) != -1) {
    switch (opt) {
      case 'n': a.N = std::atoi(optarg); break;
      case 'i': a.iters = std::atoi(optarg); break;
      case 'm': a.mode = optarg; break;
      case 'p': a.prefetch = std::atoi(optarg); break;
      case 'f': a.frac = std::atof(optarg); break;
      case 'r': a.seed = (uint64_t)std::strtoull(optarg, nullptr, 10); break;
      case 'h': usage(argv[0]); std::exit(EXIT_SUCCESS);
      default:  usage(argv[0]); std::exit(EXIT_FAILURE);
    }
  }
  return a;
}

struct DeviceCaps { int pageable=0, uses_host_pt=0, dev=0; };
static DeviceCaps query_caps() {
  DeviceCaps c{}; CHECK_CUDA(cudaGetDevice(&c.dev));
  CHECK_CUDA(cudaDeviceGetAttribute(&c.pageable, cudaDevAttrPageableMemoryAccess, c.dev));
  CHECK_CUDA(cudaDeviceGetAttribute(&c.uses_host_pt, cudaDevAttrPageableMemoryAccessUsesHostPageTables, c.dev));
  return c;
}

template <typename T> struct Buffers {
  T *hA=nullptr, *hB=nullptr; // host
  T *dA=nullptr, *dB=nullptr; // device
  size_t nbytes=0;
};

template <typename T>
static void allocate_buffers(Mode mode, size_t nElems, Buffers<T>& buf, const DeviceCaps& caps, double &t_alloc_cpu, double &t_alloc_gpu, double &t_alloc_managed, double &t_malloc, double &t_managed_memadvise) {
  buf.nbytes = nElems * sizeof(T);
  switch (mode) {
    case Mode::EXPLICIT: {
      
      double t1 = wtime();
      CHECK_CUDA(cudaMallocHost(&buf.hA, buf.nbytes));
      CHECK_CUDA(cudaMallocHost(&buf.hB, buf.nbytes));
      t_alloc_cpu += wtime() - t1;

      t1 = wtime();
      CHECK_CUDA(cudaMalloc(&buf.dA, buf.nbytes));
      CHECK_CUDA(cudaMalloc(&buf.dB, buf.nbytes));
      t_alloc_gpu += wtime() - t1;
      printf("t_alloc_cpu: %.3f ms\n", t_alloc_cpu);
      printf("t_alloc_gpu: %.3f ms\n", t_alloc_gpu);  
      break;
    }
    case Mode::UM_MIGRATE:
    case Mode::UM_MIGRATE_NO_PREFETCH:{
      double t1 = wtime();
      CHECK_CUDA(cudaMallocManaged(&buf.dA, buf.nbytes));
      CHECK_CUDA(cudaMallocManaged(&buf.dB, buf.nbytes));
      buf.hA = buf.dA; buf.hB = buf.dB; 
      t_alloc_managed += wtime() - t1;
      printf("t_alloc_managed: %.3f ms\n", t_alloc_managed);
      break;
    }
    case Mode::GH_HBM_SHARED: {
      double t1 = wtime();
      CHECK_CUDA(cudaMallocManaged(&buf.dA, buf.nbytes));
      CHECK_CUDA(cudaMallocManaged(&buf.dB, buf.nbytes));
      buf.hA = buf.dA; buf.hB = buf.dB;
      t_alloc_managed += wtime() - t1;
      printf("t_alloc_managed: %.3f ms\n", t_alloc_managed);

      t1=wtime();
      cudaMemLocation loc_dev{}; loc_dev.type = cudaMemLocationTypeDevice; loc_dev.id = caps.dev;
      CHECK_CUDA(cudaMemAdvise(buf.dA, buf.nbytes, cudaMemAdviseSetPreferredLocation, loc_dev));
      CHECK_CUDA(cudaMemAdvise(buf.dB, buf.nbytes, cudaMemAdviseSetPreferredLocation, loc_dev));
      cudaMemLocation loc_cpu{}; loc_cpu.type = cudaMemLocationTypeHost; loc_cpu.id = 0;
      CHECK_CUDA(cudaMemAdvise(buf.dA, buf.nbytes, cudaMemAdviseSetAccessedBy, loc_cpu));
      CHECK_CUDA(cudaMemAdvise(buf.dB, buf.nbytes, cudaMemAdviseSetAccessedBy, loc_cpu));
      t_managed_memadvise += wtime() - t1;
      printf("t_managed_memadvise: %.3f ms\n", t_managed_memadvise);

      break;
    }
    case Mode::GH_HMM_PAGEABLE_CUDA_INIT:
    case Mode::GH_HMM_PAGEABLE: {
      double t1 = wtime();
      buf.hA = (T*)std::malloc(buf.nbytes);
      buf.hB = (T*)std::malloc(buf.nbytes);
      if (!buf.hA || !buf.hB) { perror("malloc"); std::exit(EXIT_FAILURE); }
      buf.dA = buf.hA; buf.dB = buf.hB; 
      t_malloc += wtime() - t1;
      printf("t_malloc: %.3f ms\n", t_malloc);
      break;
    }
  }
}

template <typename T>
static void pre_kernel_setup(Mode mode, const Buffers<T>& buf, const DeviceCaps& caps, int prefetch,
                             double& ms_h2d_once, double& ms_prefetch_to_dev)
{
  ms_h2d_once = 0.0; ms_prefetch_to_dev = 0.0;
  if (mode == Mode::EXPLICIT) {
    double t0 = wtime();
    CHECK_CUDA(cudaMemcpy(buf.dA, buf.hA, buf.nbytes, cudaMemcpyHostToDevice));     
    CHECK_CUDA(cudaMemset(buf.dB, 0, buf.nbytes));
    ms_h2d_once = (wtime() - t0) * 1e3;
  } else if ((mode == Mode::UM_MIGRATE || mode == Mode::GH_HBM_SHARED) && prefetch) {
    cudaMemLocation dst_dev{}; dst_dev.type = cudaMemLocationTypeDevice; dst_dev.id = caps.dev;
    double t0 = wtime();
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dA, buf.nbytes, dst_dev, 0));
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dB, buf.nbytes, dst_dev, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    ms_prefetch_to_dev = (wtime() - t0) * 1e3;
  }
}

template <typename T>
static double checksum_host(const T* p, size_t n) {
  double s=0.0; for (size_t i=0;i<n;++i) s += (double)p[i]; return s;
}

int main(int argc, char** argv) {
  Args args = parse(argc, argv);
  Mode mode = parse_mode_str(args.mode);
  DeviceCaps caps = query_caps();
   
  double t_alloc_cpu = 0.0, t_alloc_gpu = 0.0, t_alloc_managed=0.0, t_malloc=0.0, t_managed_memadvise=0.0, t_end2end=wtime();
  const int N = args.N; const size_t elems = (size_t)N * N;
  Buffers<double> buf; allocate_buffers<double>(mode, elems, buf, caps, t_alloc_cpu, t_alloc_gpu, t_alloc_managed, t_malloc, t_managed_memadvise);

  if (mode != Mode::GH_HMM_PAGEABLE_CUDA_INIT) {
    std::mt19937_64 rng(args.seed); std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < elems; ++i) buf.hA[i] = dist(rng);
    std::fill(buf.hB, buf.hB + elems, 0.0);
  } else {
    const int threads = 256; const int blocks = int((elems + threads - 1) / threads);
    init_random_double<<<blocks, threads>>>(buf.dA, elems, args.seed);
    set_zero_double<<<blocks, threads>>>(buf.dB, elems);
    CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
  }

  
  double ms_h2d_once = 0.0, ms_prefetch_to_dev = 0.0;

  pre_kernel_setup<double>(mode, buf, caps, args.prefetch, ms_h2d_once, ms_prefetch_to_dev);
  
  int k = args.frac * N; //0-k*N: taken by CPU, the rest taken by GPU
  printf("Hybrid split: CPU columns [0, %d), GPU columns [%d, %d)\n", k, k, N);
  const int Wcpu = k, Wgpu = N - k, H = N;
  constexpr int TILEC = 32; constexpr int BRC = 8;

  //warmup: i run 2 iterations, second to return to original state
  /*const double t_warmup_start = wtime();
  {
    const int warmup_iters = 2;
    std::vector<double> w_ms_gpu(warmup_iters, 0.0), w_ms_cpu(warmup_iters, 0.0),
                        w_ms_overlap(warmup_iters, 0.0), w_ms_d2h(warmup_iters, 0.0);

    for (int it = 0; it < warmup_iters; ++it) {
      double cpu_ms_local = 0.0;
      //CPU warmup
      std::thread cpu_thr([&]{
        if (Wcpu > 0) {
          const int T = std::max(1, args.threads);
          const int chunks = std::min(T, std::max(1, Wcpu));
          const int base   = Wcpu / chunks;
          const int rem    = Wcpu % chunks;
          #pragma omp parallel for schedule(static) num_threads(T)
          for (int t = 0; t < chunks; ++t) {
            const int c0 = (t < rem) ? (t * (base + 1)) : (rem * (base + 1) + (t - rem) * base);
            const int c1 = c0 + ((t < rem) ? (base + 1) : base);
            transpose_cpu_blocked(buf.hA, buf.hB, N, c0, c1, 32);
          }
        }
      });
      //GPU warmup
      float ms_gpu_this = 0.0f;
      if (Wgpu > 0) {
        cudaEvent_t evS, evE; CHECK_CUDA(cudaEventCreate(&evS)); CHECK_CUDA(cudaEventCreate(&evE));
        dim3 block(TILEC, BRC); dim3 grid((Wgpu + TILEC - 1)/TILEC, (H + TILEC - 1)/TILEC);
        CHECK_CUDA(cudaEventRecord(evS));
        transpose_tiled_rect<double, TILEC, BRC><<<grid, block>>>(buf.dA, buf.dB, N, k, 0, Wgpu, H);
        CHECK_CUDA(cudaEventRecord(evE)); CHECK_CUDA(cudaEventSynchronize(evE));
        CHECK_CUDA(cudaEventElapsedTime(&ms_gpu_this, evS, evE));
        CHECK_CUDA(cudaEventDestroy(evS)); CHECK_CUDA(cudaEventDestroy(evE));
      }
      //join CPU, GPU done
      cpu_thr.join(); CHECK_CUDA(cudaDeviceSynchronize());
      //EXPLICIT mode, memcpy back to host (copy rows [k,N))
      if (mode == Mode::EXPLICIT) {
        const size_t row0_off = (size_t)k * (size_t)N;
        const size_t n_elems  = (size_t)(N - k) * (size_t)N;
        const void* src = (const void*)(buf.dB + row0_off);
        void*       dst = (void*)(buf.hB + row0_off);
        CHECK_CUDA(cudaMemcpy(dst, src, n_elems * sizeof(double), cudaMemcpyDeviceToHost));
      }
    }
  }
  //end warmup
  */
  
  //timed iteration
  std::vector<double> ms_gpu(args.iters, 0.0), ms_cpu(args.iters, 0.0), ms_overlap(args.iters, 0.0), ms_d2h(args.iters, 0.0);
  
  //CPU
  for (int it = 0; it < args.iters; ++it) {
   const double t_iter_start = wtime();
   double cpu_ms_local = 0.0;
   std::thread cpu_thr([&]{
   if (Wcpu > 0) {
    const double t0 = wtime();
    const int T = std::max(1, args.threads);
    //we partition the CPU columns [0, Wcpu) into 'chunks' ~ T parts
    const int chunks = std::min(T, std::max(1, Wcpu));
    const int base   = Wcpu / chunks;
    const int rem    = Wcpu % chunks;
    #pragma omp parallel for schedule(static) num_threads(T)
    for (int t = 0; t < chunks; ++t) {
      //balanced partition [0, Wcpu): each out of the T worker gets a Wcpu/T sized chunk
      const int c0 = (t < rem) ? (t * (base + 1)) : (rem * (base + 1) + (t - rem) * base);
      const int c1 = c0 + ((t < rem) ? (base + 1) : base);
      transpose_cpu_blocked(buf.hA, buf.hB, N, /*c0=*/c0, /*c1=*/c1, /*BS=*/32);
    }
    cpu_ms_local = (wtime() - t0) * 1e3;
   }
  });

    //GPU
    float ms_gpu_this = 0.0f;
    if (Wgpu > 0) {
      cudaEvent_t evS, evE; CHECK_CUDA(cudaEventCreate(&evS)); CHECK_CUDA(cudaEventCreate(&evE));
      dim3 block(TILEC, BRC); dim3 grid((Wgpu + TILEC - 1)/TILEC, (H + TILEC - 1)/TILEC);
      CHECK_CUDA(cudaEventRecord(evS));
      transpose_tiled_rect<double, TILEC, BRC><<<grid, block>>>(buf.dA, buf.dB, N, k, 0, Wgpu, H);
      CHECK_CUDA(cudaEventRecord(evE)); CHECK_CUDA(cudaEventSynchronize(evE));
      CHECK_CUDA(cudaEventElapsedTime(&ms_gpu_this, evS, evE));
      CHECK_CUDA(cudaEventDestroy(evS)); CHECK_CUDA(cudaEventDestroy(evE));
    }

    //join CPU, finish GPU 
    cpu_thr.join(); CHECK_CUDA(cudaDeviceSynchronize());

    //EXPLICIT mode, memcpy back to host (copy rows [k,N) that were handled by GPU)
    if (mode == Mode::EXPLICIT) {
      double t0 = wtime();
      const size_t row0_off = (size_t)k * (size_t)N;
      const size_t n_elems  = (size_t)(N - k) * (size_t)N;
      const void* src = (const void*)(buf.dB + row0_off);
      void*       dst = (void*)(buf.hB + row0_off);
      CHECK_CUDA(cudaMemcpy(dst, src, n_elems * sizeof(double), cudaMemcpyDeviceToHost));
      ms_d2h[it] = (wtime() - t0) * 1e3;
    }

    //per iteration
    ms_cpu[it]    = cpu_ms_local;          
    ms_gpu[it]    = (double)ms_gpu_this;   
    ms_overlap[it]= (wtime() - t_iter_start) * 1e3; //compute end to end for this iteration
  }
  
   

  //UM migrate back for CPU read timing
  /*double ms_um_to_cpu = 0.0;
  if (mode == Mode::UM_MIGRATE) {
    cudaMemLocation dst_cpu{}; dst_cpu.type = cudaMemLocationTypeHost; dst_cpu.id = 0;
    double t0 = wtime();
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dB, buf.nbytes, dst_cpu, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    ms_um_to_cpu = (wtime() - t0) * 1e3;
  }*/

  //host checksum read (captures migration costs for no-prefetch/HMM)
  //double sumB = checksum_host(buf.hB, elems);
  double maxerr = max_abs_diff_transpose(buf.hA, buf.hB, N);

  //printf("CPU checksum(B)=%.6e\n", sumB);
  printf("Max |B - A^T| = %.3e  => %s\n", maxerr, (maxerr < 1e-9 ? "OK" : "MISMATCH"));


  double t1, t_dealloc_cpu = 0.0, t_dealloc_gpu=0.0, t_dealloc_managed = 0.0, t_dealloc_malloc = 0.0;
  switch (mode) {
    case Mode::EXPLICIT:
      t1 = wtime();
      CHECK_CUDA(cudaFree(buf.dA)); 
      CHECK_CUDA(cudaFree(buf.dB));
      t_dealloc_gpu += wtime() - t1;
      printf("t_dealloc_gpu: %.3f ms\n", t_dealloc_gpu);
      t1 = wtime();
      CHECK_CUDA(cudaFreeHost(buf.hA)); 
      CHECK_CUDA(cudaFreeHost(buf.hB));
      t_dealloc_cpu += wtime() - t1;
      printf("t_dealloc_cpu: %.3f ms\n", t_dealloc_cpu);
      break;
    case Mode::UM_MIGRATE:
    case Mode::GH_HBM_SHARED:
    case Mode::UM_MIGRATE_NO_PREFETCH:
    case Mode::GH_HBM_SHARED_NO_PREFETCH:
      t1 = wtime();
      CHECK_CUDA(cudaFree(buf.dA)); CHECK_CUDA(cudaFree(buf.dB));
      t_dealloc_managed += wtime() - t1;
      printf("t_dealloc_managed: %.3f ms\n", t_dealloc_managed);
      break;
    case Mode::GH_HMM_PAGEABLE_CUDA_INIT:
    case Mode::GH_HMM_PAGEABLE:
      t1 = wtime();
      std::free(buf.hA); std::free(buf.hB); 
      t_dealloc_malloc += wtime() - t1;
      printf("t_dealloc_malloc: %.3f ms\n", t_dealloc_malloc);
      break;
  }


  //END OF FULL END TO END RUN
  const double ms_full_e2e = (wtime() - t_end2end) * 1e3; 

  // double t_verify0 = wtime();
  //double maxerr = max_abs_diff_transpose(buf.hA, buf.hB, N);
  // double ms_verify = (wtime() - t_verify0) * 1e3;

  //prints
  //printf("Pre Kernel Costs: %.3f ms \n", pre_kernel_ms);


  double total_d2h = std::accumulate(ms_d2h.begin(), ms_d2h.end(), 0.0);
  double total_gpu = std::accumulate(ms_gpu.begin(), ms_gpu.end(), 0.0);
  double total_cpu = std::accumulate(ms_cpu.begin(), ms_cpu.end(), 0.0);

  if (mode == Mode::UM_MIGRATE || mode == Mode::GH_HBM_SHARED && args.prefetch)
    printf("t_managed_prefetch: %.3f ms\n", ms_prefetch_to_dev);

  if (mode == Mode::GH_HBM_SHARED)
    printf("t_managed_memadvise: %.3f ms\n", ms_prefetch_to_dev);

  if (mode == Mode::EXPLICIT)
    printf("t_transfers: %.3f ms\n", ms_h2d_once);
  if (mode == Mode::EXPLICIT && Wgpu>0)
    printf("t_transfers: %.3f ms\n", total_d2h + ms_h2d_once);

  //if (Wgpu>0) printf("GPU Kernel avg: %.3f ms\n", ms_gpu_avg);
  //else        printf("GPU Kernel avg: N/A (no GPU work)\n");
  //if (Wcpu>0) printf("CPU compute:    %.3f ms\n", ms_cpu_avg);
  //else        printf("CPU compute:    N/A (no CPU work)\n");
  if (Wgpu>0) printf("t_gpu_computation: %.3f ms\n", total_gpu);
  if (Wcpu>0) printf("t_cpu_computation:    %.3f ms\n", total_cpu);

  printf("t_end_2_end:  %.3f ms\n", ms_full_e2e);
  printf("t_other:  %.3f ms\n", (ms_full_e2e - total_gpu - total_cpu - total_d2h - ms_h2d_once - ms_prefetch_to_dev - t_dealloc_cpu - t_dealloc_gpu - t_dealloc_managed - t_dealloc_malloc - t_alloc_cpu - t_alloc_gpu - t_alloc_managed - t_malloc - t_managed_memadvise));


  /*printf("CPU checksum(B)=%.6e, CPU read time: %.3f ms\n", sumB, ms_host_read);
  printf("Max |B - A^T| = %.3e  => %s\n", maxerr, (maxerr < 1e-9 ? "OK" : "MISMATCH"));
  */
  /*switch (mode) {
    case Mode::EXPLICIT:
      CHECK_CUDA(cudaFree(buf.dA)); CHECK_CUDA(cudaFree(buf.dB));
      CHECK_CUDA(cudaFreeHost(buf.hA)); CHECK_CUDA(cudaFreeHost(buf.hB));
      break;
    case Mode::UM_MIGRATE:
    case Mode::GH_HBM_SHARED:
    case Mode::UM_MIGRATE_NO_PREFETCH:
    case Mode::GH_HBM_SHARED_NO_PREFETCH:
      CHECK_CUDA(cudaFree(buf.dA)); CHECK_CUDA(cudaFree(buf.dB));
      break;
    case Mode::GH_CPU_SHARED:
      CHECK_CUDA(cudaFreeHost(buf.hA)); CHECK_CUDA(cudaFreeHost(buf.hB));
      break;
    case Mode::GH_HMM_PAGEABLE_CUDA_INIT:
    case Mode::GH_HMM_PAGEABLE:
      std::free(buf.hA); std::free(buf.hB); break;
  }*/


  return 0;
}

