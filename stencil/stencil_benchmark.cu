#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <unistd.h>   // getopt
#include <sys/time.h> // gettimeofday
#include <numeric>    // accumulate

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


enum class Mode {
  EXPLICIT, //pinned host + device buffers, manual H2D/D2H copies
  EXPLICIT_ASYNC, 
  UM_MIGRATE, //managed memory with prefetching to GPU and back to CPU at the end
  GH_HBM_SHARED, // managed memory with prefetching to GPU; results migrate on demand to CPU
  GH_CPU_SHARED, //pinned host memory mapped into device address via cudaHostGetDevicePointer(GPU accesses not pageable DDR)
  GH_HMM_PAGEABLE, //malloc(pageable) pointer and the GPU can access those directly, first touch on CPU
  GH_HMM_PAGEABLE_GPU_INIT, //first touch on GPU
  UM_MIGRATE_NO_PREFETCH, //no prefetching to the GPU 
  GH_HBM_SHARED_NO_PREFETCH
};

static Mode parse_mode_str(const std::string& s) {
  if (s == "explicit")                  return Mode::EXPLICIT;
  if (s == "explicit_async")                  return Mode::EXPLICIT_ASYNC;
  if (s == "um_migrate")                return Mode::UM_MIGRATE;
  if (s == "gh_hbm_shared")             return Mode::GH_HBM_SHARED;
  if (s == "gh_cpu_shared")             return Mode::GH_CPU_SHARED;
  if (s == "gh_hmm_pageable")           return Mode::GH_HMM_PAGEABLE;
  if (s == "gh_hmm_pageable_gpu_init")  return Mode::GH_HMM_PAGEABLE_GPU_INIT;
  if (s == "um_migrate_no_prefetch")    return Mode::UM_MIGRATE_NO_PREFETCH;
  if (s == "gh_hbm_shared_no_prefetch") return Mode::GH_HBM_SHARED_NO_PREFETCH;
  fprintf(stderr, "Unknown --mode=%s\n", s.c_str());
  std::exit(EXIT_FAILURE);
}


struct Args {
  int N = 4096;           // grid dimension N x N
  int iters = 10;         // iterations
  std::string mode = "explicit";
  int prefetch = 1;       // only relevant to UM_MIGRATE / GH_HBM_SHARED 
  uint64_t seed = 12345;
  int steps = 1;          // inner stencil time-steps per iteration (ping-pong buffers)
};

static void usage(const char* prog) {
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  -n INT    problem size N (default 4096; domain is N x N)\n"
    "  -i INT    iterations (default 10)\n"
    "  -t INT    stencil time-steps per iteration (default 1)\n"
    "  -m STR    mode: explicit | explicit_async | um_migrate | gh_hbm_shared | gh_cpu_shared | gh_hmm_pageable |\n"
    "            gh_hmm_pageable_gpu_init | um_migrate_no_prefetch | gh_hbm_shared_no_prefetch (default explicit)\n"
    "  -p 0|1    prefetch for UM/GH_HBM_SHARED (default 1)\n"
    "  -r UINT64 RNG seed (default 12345)\n"
    "  -h        help\n", prog);
}

static Args parse(int argc, char** argv) {
  Args a;
  int opt;
  while ((opt = getopt(argc, argv, "n:i:t:m:p:r:h")) != -1) {
    switch (opt) {
      case 'n': a.N = std::atoi(optarg); break;
      case 'i': a.iters = std::atoi(optarg); break;
      case 't': a.steps = std::atoi(optarg); break;
      case 'm': a.mode = optarg; break;
      case 'p': a.prefetch = std::atoi(optarg); break;
      case 'r': a.seed = (uint64_t)std::strtoull(optarg, nullptr, 10); break;
      case 'h': usage(argv[0]); std::exit(EXIT_SUCCESS);
      default:  usage(argv[0]); std::exit(EXIT_FAILURE);
    }
  }
  if (a.N <= 0 || a.iters <= 0 || a.steps <= 0) {
    fprintf(stderr, "Error: N, iters, and steps must be > 0\n");
    std::exit(EXIT_FAILURE);
  }
  return a;
}

//fast 64-bit integer hash used as a simple PRNG (SplitMix64) on the device
__device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

//converts a uint64_t to a uniform double in [0,1) using top 53 bits
__device__ inline double u01_from_u64(uint64_t x) {
  const double inv2_53 = 1.0 / 9007199254740992.0; // 2^53
  return double(x >> 11) * inv2_53;
}

//initializes a[0..n-1] with pseudo-random doubles in [-1,1)
__global__ void init_random_double(double* a, size_t n, uint64_t seed) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; //one thread per element
  if (i < n) {
    uint64_t r = splitmix64(seed + i);
    double u = u01_from_u64(r);
    a[i] = 2.0 * u - 1.0; //[-1,1)
  }
}

//device-side memset for doubles
__global__ void set_zero_double(double* a, size_t n) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) a[i] = 0.0;
}

//2D 5-point stencil; borders: copy input to output
__global__ void stencil5_2d(const double* __restrict__ in, double* __restrict__ out,
                            int N)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x; //column
  int y = blockIdx.y * blockDim.y + threadIdx.y; //row
  if (x >= N || y >= N) return;

  const size_t idx = (size_t)y * N + x; //map the (x,y) to the flat idx 

  if (x == 0 || y == 0 || x == N-1 || y == N-1) {
    out[idx] = in[idx]; //border policy: copy input to output unchanged on boundary cells
    return;
  }

  //interior cells
  const size_t idxN = (size_t)(y-1)*N + x;
  const size_t idxS = (size_t)(y+1)*N + x;
  const size_t idxW = (size_t)y*N + (x-1);
  const size_t idxE = (size_t)y*N + (x+1);

  double v = 0.25 * (in[idxN] + in[idxS] + in[idxW] + in[idxE]);
  out[idx] = v;
}


struct DeviceCaps {
  int pageable = 0;
  int uses_host_pt = 0;
  int dev = 0;
};

static DeviceCaps query_caps() {
  DeviceCaps c{};
  CHECK_CUDA(cudaGetDevice(&c.dev));
  CHECK_CUDA(cudaDeviceGetAttribute(&c.pageable, cudaDevAttrPageableMemoryAccess, c.dev));
  CHECK_CUDA(cudaDeviceGetAttribute(&c.uses_host_pt, cudaDevAttrPageableMemoryAccessUsesHostPageTables, c.dev));
  return c;
}

template <typename T>
struct Buffers { //track host & device buffers(pointers) size 
  T *hIn=nullptr, *hOut=nullptr;
  T *dIn=nullptr, *dOut=nullptr;
  size_t nbytes=0;
};

//allocation style for each mode
template <typename T>
static void allocate_buffers(Mode mode, size_t nElems, Buffers<T>& buf, const DeviceCaps& caps) {
  buf.nbytes = nElems * sizeof(T);
  switch (mode) {
    case Mode::EXPLICIT: {
      CHECK_CUDA(cudaMallocHost(&buf.hIn,  buf.nbytes));
      CHECK_CUDA(cudaMallocHost(&buf.hOut, buf.nbytes));
      CHECK_CUDA(cudaMalloc(&buf.dIn,  buf.nbytes));
      CHECK_CUDA(cudaMalloc(&buf.dOut, buf.nbytes));
      break;
    }
    case Mode::EXPLICIT_ASYNC: {
      CHECK_CUDA(cudaMallocHost(&buf.hIn,  buf.nbytes));
      CHECK_CUDA(cudaMallocHost(&buf.hOut, buf.nbytes));
      CHECK_CUDA(cudaMallocAsync(&buf.dIn,  buf.nbytes, 0));
      CHECK_CUDA(cudaMallocAsync(&buf.dOut, buf.nbytes, 0));
      break;
    }
    case Mode::UM_MIGRATE:
    case Mode::GH_HBM_SHARED:
    case Mode::UM_MIGRATE_NO_PREFETCH:
    case Mode::GH_HBM_SHARED_NO_PREFETCH: {
      CHECK_CUDA(cudaMallocManaged(&buf.dIn,  buf.nbytes));
      CHECK_CUDA(cudaMallocManaged(&buf.dOut, buf.nbytes));
      buf.hIn  = buf.dIn;  //same pointers for GPU nad CPU
      buf.hOut = buf.dOut;
      break;
    }
    case Mode::GH_CPU_SHARED: { //page-locked memory allocated on host, zero copy - use of cudaHostGetDevicePointer via NVLink
      CHECK_CUDA(cudaMallocHost(&buf.hIn,  buf.nbytes));
      CHECK_CUDA(cudaMallocHost(&buf.hOut, buf.nbytes));
      CHECK_CUDA(cudaHostGetDevicePointer(&buf.dIn,  buf.hIn,  0));
      CHECK_CUDA(cudaHostGetDevicePointer(&buf.dOut, buf.hOut, 0));
      break;
    }
    case Mode::GH_HMM_PAGEABLE_GPU_INIT:
    case Mode::GH_HMM_PAGEABLE: { //system-wide page table, pagable memory on host through malloc, DMA by GPU via NVLink
      if (!caps.pageable) {
        fprintf(stderr, "ERROR: Device lacks cudaDevAttrPageableMemoryAccess; HMM pageable not supported.\n");
        std::exit(EXIT_FAILURE);
      }
      buf.hIn  = (T*)std::malloc(buf.nbytes);
      buf.hOut = (T*)std::malloc(buf.nbytes);
      if (!buf.hIn || !buf.hOut) { perror("malloc"); std::exit(EXIT_FAILURE); }
      buf.dIn  = buf.hIn;   //same raw pointers (HMM)
      buf.dOut = buf.hOut;
      printf("HMM pageable supported (uses_host_page_tables=%d)\n", caps.uses_host_pt);
      break;
    }
  }
}

template <typename T> //prefetching for some of the modes
static void pre_kernel_setup(Mode mode, const Buffers<T>& buf, const DeviceCaps& caps, int do_prefetch) {
  if (mode == Mode::EXPLICIT || mode == Mode::EXPLICIT_ASYNC) {
    CHECK_CUDA(cudaMemcpy(buf.dIn, buf.hIn, buf.nbytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(buf.dOut, 0, buf.nbytes));
  } else if ((mode == Mode::UM_MIGRATE || mode == Mode::GH_HBM_SHARED) && do_prefetch) {
    // FIX: Updated to use cudaMemLocation and stream argument (0)
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dIn,  buf.nbytes, {cudaMemLocationTypeDevice, caps.dev}, 0));
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dOut, buf.nbytes, {cudaMemLocationTypeDevice, caps.dev}, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
  } else {
    //NO_PREFETCH/GH_CPU_SHARED/GH_HMM_PAGEABLE: nothing here
  }
}

template <typename T> //migrate results back to the CPU so the CPU checksum doesn't trigger page faults, time the prefetching
static void post_kernel_maybe_prefetch_back(Mode mode, const Buffers<T>& buf) {
  if (mode == Mode::UM_MIGRATE) {
    double t0 = wtime();
    // FIX: Updated to use cudaMemLocationTypeHost and stream argument (0)
    CHECK_CUDA(cudaMemPrefetchAsync(buf.dOut, buf.nbytes, {cudaMemLocationTypeHost, 0}, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    double ms = (wtime() - t0) * 1e3;
    printf("UM prefetch-to-CPU: %.3f ms (%.2f GB/s logical)\n", ms, double(buf.nbytes) / 1e6 / ms);
  }
}

template <typename T> //manual copies back to the CPU(D2H) after the kernel is complete
static void post_kernel_copy_back_if_needed(Mode mode, const Buffers<T>& buf) {
  if (mode == Mode::EXPLICIT || mode == Mode::EXPLICIT_ASYNC) {
    double t0 = wtime();
    CHECK_CUDA(cudaMemcpy(buf.hOut, buf.dOut, buf.nbytes, cudaMemcpyDeviceToHost));
    double ms = (wtime() - t0) * 1e3;
    printf("D2H memcpy: %.3f ms (%.2f GB/s)\n", ms, double(buf.nbytes) / 1e6 / ms);
  }
}

template <typename T>
static void free_buffers(Mode mode, Buffers<T>& buf) {
  switch (mode) {
    case Mode::EXPLICIT:
      CHECK_CUDA(cudaFree(buf.dIn));
      CHECK_CUDA(cudaFree(buf.dOut));
      CHECK_CUDA(cudaFreeHost(buf.hIn));
      CHECK_CUDA(cudaFreeHost(buf.hOut));
      break;
    case Mode::EXPLICIT_ASYNC:
      CHECK_CUDA(cudaFreeAsync(buf.dIn, 0));
      CHECK_CUDA(cudaFreeAsync(buf.dOut, 0));
      CHECK_CUDA(cudaFreeHost(buf.hIn));
      CHECK_CUDA(cudaFreeHost(buf.hOut));
      break;

    case Mode::UM_MIGRATE:
    case Mode::GH_HBM_SHARED:
    case Mode::UM_MIGRATE_NO_PREFETCH:
    case Mode::GH_HBM_SHARED_NO_PREFETCH:
      CHECK_CUDA(cudaFree(buf.dIn));
      CHECK_CUDA(cudaFree(buf.dOut));
      break;
    case Mode::GH_CPU_SHARED:
      CHECK_CUDA(cudaFreeHost(buf.hIn));
      CHECK_CUDA(cudaFreeHost(buf.hOut));
      break;
    case Mode::GH_HMM_PAGEABLE_GPU_INIT:
    case Mode::GH_HMM_PAGEABLE:
      std::free(buf.hIn);
      std::free(buf.hOut);
      break;
  }
}


template <typename T> //check correctness and also a way to check CPU read bandwidth
static double checksum(const T* p, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) s += (double)p[i];
  return s;
}

// Function to calculate average of vector<float>
auto average_float = [](const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += x;
  return (v.empty() ? 0.0 : s / double(v.size()));
};


int main(int argc, char** argv) {
  Args args = parse(argc, argv);
  Mode mode = parse_mode_str(args.mode);
  DeviceCaps caps = query_caps();

  const int N = args.N;
  const size_t elems = (size_t)N * N;
  const size_t bytes = elems * sizeof(double);

  printf("stencil 5-point | N=%d (%.3f MiB), iters=%d, steps/iter=%d, mode=%s, prefetch=%d, seed=%llu\n",
         N, double(bytes)/(1024.0*1024.0), args.iters, args.steps,
         args.mode.c_str(), args.prefetch, (unsigned long long)args.seed);

  Buffers<double> buf;
  allocate_buffers<double>(mode, elems, buf, caps);

  //initialization for each mode
  if (mode == Mode::GH_HMM_PAGEABLE_GPU_INIT) {
    //initialize on GPU into pageable host memory via HMM
    const int threads = 256;
    const int blocks  = int((elems + threads - 1) / threads);
    init_random_double<<<blocks, threads>>>(buf.dIn, elems, args.seed);
    set_zero_double<<<blocks, threads>>>(buf.dOut, elems);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  } else {
    //CPU init path for all other modes
    std::mt19937_64 rng(args.seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < elems; ++i) buf.hIn[i] = dist(rng);
    std::fill(buf.hOut, buf.hOut + elems, 0.0);
  }

  //prefetch/copies before kernel
  pre_kernel_setup<double>(mode, buf, caps, args.prefetch);

  //launch config----------
  dim3 block(32, 8);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  // =======================
  //     PER-STEP TIMING LOGIC
  // =======================
  std::vector<float> total_times(args.iters, 0.0f);
  std::vector<float> step1_times(args.iters, 0.0f);
  std::vector<float> warm_avg_times(args.iters, 0.0f); 

  for (int it = 0; it < args.iters; ++it) {
    // set starting endpoints for this iteration
    const double* in  = buf.dIn;
    double* out = buf.dOut;
    const int S = args.steps;

    // Create a chain of events: ev[0] before step 1, ev[k] after step k
    std::vector<cudaEvent_t> ev(S + 1);
    for (auto& e : ev) CHECK_CUDA(cudaEventCreate(&e));

    // Record start event (before step 1)
    CHECK_CUDA(cudaEventRecord(ev[0]));

    // Perform S steps, recording after each step; swap in/out each time (ping-pong)
    for (int t = 0; t < S; ++t) {
      stencil5_2d<<<grid, block>>>(in, out, N);
      CHECK_CUDA(cudaEventRecord(ev[t + 1])); // after step t+1

      // swap for ping-pong
      const double* tmp_in = in;
      in  = out;
      out = (double*)tmp_in;
    }

    // One sync at the end ensures all prior work/events are complete
    CHECK_CUDA(cudaEventSynchronize(ev[S]));

    // Per-step times: Δt_k = ev[k] - ev[k-1]
    std::vector<float> step_ms(S, 0.0f);
    for (int k = 1; k <= S; ++k) {
      CHECK_CUDA(cudaEventElapsedTime(&step_ms[k - 1], ev[k - 1], ev[k]));
    }

    for (auto& e : ev) CHECK_CUDA(cudaEventDestroy(e));

    // Save per-iteration stats
    float total = std::accumulate(step_ms.begin(), step_ms.end(), 0.0f);
    total_times[it] = total;
    step1_times[it] = step_ms[0];

    if (S > 1) {
      float warm_sum = std::accumulate(step_ms.begin() + 1, step_ms.end(), 0.0f);
      warm_avg_times[it] = warm_sum / float(S - 1);
    } else {
      warm_avg_times[it] = 0.0f;
    }
    
    // If steps is odd, ensure buf.dOut points to the latest result (same intent as original code)
    if ((S & 1) == 1) {
      //swap buffer roles
      double* tmp = buf.dOut;
      buf.dOut = (double*)buf.dIn;
      buf.dIn  = tmp;
    }
  }

  const double avg_total_per_iter_ms = average_float(total_times);
  const double avg_step1_ms          = average_float(step1_times);
  const double avg_warm_ms           = average_float(warm_avg_times);
  /*
  printf("Kernel avg (total per iteration over %d step(s)): %.3f ms\n",
           args.steps, avg_total_per_iter_ms);
  printf("  ├─ Step 1 (cold): %.3f ms\n", avg_step1_ms);
  if (args.steps > 1)
    printf("  └─ Steps 2..%d (warm, per-step avg): %.3f ms\n", args.steps, avg_warm_ms);
  else
    printf("  └─ No warm steps (steps=1)\n");
  */


  
  printf("Kernel avg (total per iteration over %d step(s)): %.3f ms\n",
           args.steps, avg_total_per_iter_ms);
  printf("  [Step 1 cold]: %.3f ms\n", avg_step1_ms); // <- MUST BE THIS FORMAT
  if (args.steps > 1)
    printf("  [Steps 2..%d warm avg]: %.3f ms\n", args.steps, avg_warm_ms); // <- MUST BE THIS FORMAT
  else
    printf("  [No warm steps (steps=1)]\n");
  





  fflush(stdout); // Ensure kernel times are printed immediately

  //managed migrate back (only UM_MIGRATE)
  post_kernel_maybe_prefetch_back<double>(mode, buf);

  //explicit copy back if needed
  post_kernel_copy_back_if_needed<double>(mode, buf);

  //CPU checksum timing
  double t0 = wtime();
  double sum = checksum<double>(buf.hOut, elems);
  double read_ms = (wtime() - t0) * 1e3;
  printf("CPU checksum(out)=%.6e, CPU read time: %.3f ms (%.2f GB/s logical)\n",
         sum, read_ms, double(bytes) / 1e6 / read_ms);
         
  free_buffers<double>(mode, buf);
  return 0;
}
