#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/NumericLimits.cuh>

#include <THC/THC.h>
#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>

using Tensor = at::Tensor;
using TensorList = at::TensorList;
using ScalarType = at::ScalarType;
using at::acc_type;

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxForwardEpilogue {
  __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : logsum(max_input + std::log(sum)) {}

  __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_log_sum_exp)
    : logsum(max_log_sum_exp) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(input - logsum);
}

  const AccumT logsum;
};

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxBackwardEpilogue {
  __device__ __forceinline__ LogSoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

  __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(gradOutput - std::exp(static_cast<AccumT>(output)) * sum);
  }

  const AccumT sum;
};

template<typename T, typename AccumT, typename OutT>
struct SoftMaxForwardEpilogue {
  __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : max_input(max_input)
    , sum(sum) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(std::exp(input - max_input) / sum);
  }

  const AccumT max_input;
  const AccumT sum;
};

template<typename T, typename AccumT, typename OutT>
struct SoftMaxBackwardEpilogue {
  __device__ __forceinline__ SoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

  // XXX: gradOutput that we get here is really gradOutput * output
  // Look for cmul in SoftMax_updateGradInput
  __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(gradOutput - output * sum);
  }

  const AccumT sum;
};



const int max_threads = 1024;

inline dim3 SoftMax_getBlockSize(int ILP, uint64_t dim_size) {
  uint64_t block_size = 1;
  uint64_t max_block_size = std::min(dim_size / ILP, static_cast<uint64_t>(max_threads));
  while (block_size < max_block_size) block_size *= 2;
  // Launch at least a single warp - the kernel assumes that.
  block_size = std::max(block_size, static_cast<uint64_t>(32));
  return dim3(block_size);
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};


////////////////////////////////////////////////////////////////////////////////
// Regular kernel (fast when dim_size is large; requires inner_size == 1)
////////////////////////////////////////////////////////////////////////////////


template <typename T, typename AccumT>
struct MaxFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return ::max(max, (AccumT)v);
  }
};

template<typename T, typename AccumT>
struct AddFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + v;
  }
};

template<typename T, typename AccumT>
struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(AccumT v)
    : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + std::exp(v - max_k);
  }

  const AccumT max_k;
};

template <template<typename> class Reduction, typename AccumT>
__device__ __forceinline__ AccumT
blockReduce(AccumT* smem, AccumT val,
            const Reduction<AccumT>& r,
            AccumT defaultVal)
{
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  AccumT warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  uint32_t mask = (((uint64_t)1) << (blockDim.x / 32)) - 1;
  if (threadIdx.x < 32) {
    int lane = threadIdx.x % 32;
    if (lane < blockDim.x / 32) {
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        warpVal = r(warpVal, smem[lane * 32 + i]);
      }
      __syncwarp(mask);
      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal = defaultVal;

  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / 32; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

template <template<typename> class Reduction1, template<typename> class Reduction2, typename AccumT>
__device__ __forceinline__ void
blockReduce(AccumT* smem,
            AccumT* reducVal1,
            AccumT val1,
            const Reduction1<AccumT>& r1,
            AccumT defaultVal1,
            AccumT* reducVal2,
            AccumT val2,
            const Reduction2<AccumT>& r2,
            AccumT defaultVal2)
{
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();

  smem[threadIdx.x] = val1;
  smem[blockDim.x + threadIdx.x] = val2;

  __syncthreads();

  AccumT warpVal1 = defaultVal1;
  AccumT warpVal2 = defaultVal2;

  // First warp will perform per-warp reductions for the remaining warps
  uint32_t mask = (((uint64_t)1) << (blockDim.x / 32)) - 1;
  if (threadIdx.x < 32) {
    int lane = threadIdx.x % 32;
    if (lane < blockDim.x / 32) {
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        warpVal1 = r1(warpVal1, smem[lane * 32 + i]);
        warpVal2 = r2(warpVal2, smem[lane * 32 + i + blockDim.x]);
      }
      __syncwarp(mask);
      smem[lane] = warpVal1;
      smem[lane + blockDim.x] = warpVal2;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal1 = defaultVal1;
  AccumT blockVal2 = defaultVal2;

  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / 32; ++i) {
      blockVal1 = r1(blockVal1, smem[i]);
      blockVal2 = r2(blockVal2, smem[i + blockDim.x]);
    }
    smem[0] = blockVal1;
    smem[blockDim.x] = blockVal2;
  }

  // Sync and broadcast
  __syncthreads();
  *reducVal1 = smem[0];
  *reducVal2 = smem[blockDim.x];
  __syncthreads();
}

template <template<typename, typename> class Reduction, int ILP, typename T, typename AccumT>
__device__ __forceinline__ AccumT
ilpReduce(T* data,
          int size,
          const Reduction<T, AccumT>& r,
          AccumT defaultVal)
{
  AccumT threadVal = defaultVal;
  int offset = threadIdx.x;

  int last = size % (ILP * blockDim.x);

  // Body (unroll by ILP times)
  for (; offset < size - last; offset += blockDim.x * ILP) {
    T tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      tmp[j] = data[offset + j * blockDim.x];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      threadVal = r(threadVal, tmp[j]);
  }

  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);

  return threadVal;
}

template <template<typename, typename> class Reduction1, template<typename, typename> class Reduction2, int ILP, typename T, typename AccumT>
__device__ __forceinline__ void
ilpReduce(T* data,
          int size,
          AccumT* reducVal1,
          const Reduction1<T, AccumT>& r1,
          AccumT defaultVal1,
          AccumT* reducVal2,
          const Reduction2<T, AccumT>& r2,
          AccumT defaultVal2)
{
  AccumT threadVal1 = defaultVal1;
  AccumT threadVal2 = defaultVal2;
  int offset = threadIdx.x;

  int last = size % (ILP * blockDim.x);

  // Body (unroll by ILP times)
  for (; offset < size - last; offset += blockDim.x * ILP) {
    T tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      tmp[j] = data[offset + j * blockDim.x];

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      threadVal1 = r1(threadVal1, tmp[j]);
      threadVal2 = r2(threadVal2, tmp[j]);
    }
  }

  // Epilogue
  for (; offset < size; offset += blockDim.x) {
    threadVal1 = r1(threadVal1, data[offset]);
    threadVal2 = r2(threadVal2, data[offset]);
  }

  *reducVal1 = threadVal1;
  *reducVal2 = threadVal2;
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxXEntropyForward(
    accscalar_t *losses,
    outscalar_t *max_log_sum_exp,
    scalar_t *input,
    int64_t *labels,
    int64_t classes,
    const float smoothing)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  //output += blockIdx.x * classes;

  int64_t label = labels[blockIdx.x];

  // find the max and sum
  accscalar_t threadMax, threadSum, max_k, sum_k;
  ilpReduce<MaxFloat, AddFloat, ILP, scalar_t, accscalar_t>(
      input, classes,
      &threadMax, MaxFloat<scalar_t, accscalar_t>(),
      -at::numeric_limits<accscalar_t>::max(),
      &threadSum, AddFloat<scalar_t, accscalar_t>(),
      static_cast<accscalar_t>(0));
  blockReduce<Max, Add, accscalar_t>(
      sdata,
      &max_k, threadMax, Max<accscalar_t>(),
      -at::numeric_limits<accscalar_t>::max(),
      &sum_k, threadSum, Add<accscalar_t>(),
      static_cast<accscalar_t>(0));

  // reduce all values
  accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
      input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
  accscalar_t sumAll = blockReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);

  // calculate per element loss with label smoothing
  // reserve max + log_sum_exp for bprop
  if (threadIdx.x == 0) {
    accscalar_t log_prob = epilogue(static_cast<accscalar_t>(input[label]));
    losses[blockIdx.x] = (max_k + std::log(sumAll) - sum_k / classes) \
      * smoothing - log_prob * (1 - smoothing);
    max_log_sum_exp[blockIdx.x] = max_k + std::log(sumAll);
  }
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxXEntropyBackward(
    scalar_t *gradInput,
    scalar_t *logits,
    outscalar_t *max_log_sum_exp,
    outscalar_t *gradOutput,
    int64_t *labels,
    const float smoothing,
    int classes)
{
  gradInput += blockIdx.x * classes;
  logits += blockIdx.x * classes;

  float smooth_positives = 1.0 - smoothing;
  float smooth_negatives = smoothing / classes;
  outscalar_t tmpGradOutput = gradOutput[blockIdx.x];
  int64_t label = labels[blockIdx.x];
  outscalar_t coeff = max_log_sum_exp[blockIdx.x];

  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    accscalar_t tmpLogits[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmpLogits[j] = static_cast<accscalar_t>(logits[offset + j * blockDim.x]);
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      gradInput[offset + j * blockDim.x] = tmpGradOutput * (
         std::exp(tmpLogits[j] - coeff) - static_cast<outscalar_t>(
         (offset + j * blockDim.x == label) ? 1 : 0) *
         smooth_positives - smooth_negatives);
  }

  for (; offset < classes; offset += blockDim.x)
    gradInput[offset] = tmpGradOutput * (std::exp(
        static_cast<accscalar_t>(logits[offset]) - coeff) - 
        static_cast<outscalar_t>((offset == label) ? 1 : 0) *
        smooth_positives - smooth_negatives);
}






template<template<typename, typename, typename> class Epilogue>
std::vector<Tensor> host_softmax_xentropy(
        const Tensor & input_,
        const Tensor & labels_,
        const float smoothing,
        const bool half_to_float){
  if (half_to_float) AT_ASSERTM(input_.type().scalarType() == ScalarType::Half,"conversion is supported for Half type only");
  AT_ASSERTM(labels_.type().scalarType() == ScalarType::Long,"Label type should be CUDA Long");
  auto input = input_.contiguous();
  Tensor max_log_sum_exp = at::empty_like(labels_, half_to_float ? input.options().dtype(ScalarType::Float) : input.options());
  static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
  Tensor losses = at::empty_like(labels_, input_.options().dtype(ScalarType::Float));
  if (input.dim() == 0) input = input.view(1);
  int64_t dim = 1;
  AT_ASSERTM(input.dim() == 2, "Currently only 2 dim input supported");
  AT_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);

  if (input.numel() > 0) {
    int64_t inner_size = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int64_t i = 0; i < dim; ++i)
      outer_size *= input.size(i);
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);
    // This kernel spawns a block per each element in the batch.
    // XXX: it assumes that inner_size == 1
    AT_CHECK(inner_size == 1, "Currently only inner size 1 supported");

    const int ILP = 2;
    dim3 grid(outer_size);
    dim3 block = SoftMax_getBlockSize(ILP, dim_size);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "host_softmax_xentropy", [&] {
    using accscalar_t = acc_type<scalar_t, true>;
    if (!half_to_float) {
        cunn_SoftMaxXEntropyForward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
          <<<grid, block, 2 * block.x * sizeof(accscalar_t), stream>>>(
            losses.data<accscalar_t>(), max_log_sum_exp.data<scalar_t>(),
            input.data<scalar_t>(), labels_.data<int64_t>(),
            dim_size, smoothing
        );
    } else {
        cunn_SoftMaxXEntropyForward<ILP, scalar_t, accscalar_t, accscalar_t, Epilogue>
          <<<grid, block, 2 * block.x * sizeof(accscalar_t), stream>>>(
            losses.data<accscalar_t>(), max_log_sum_exp.data<accscalar_t>(),
            input.data<scalar_t>(), labels_.data<int64_t>(),
            dim_size, smoothing
        );
    }
    });

    THCudaCheck(cudaGetLastError());
  }
  std::vector<at::Tensor> ret = {losses, max_log_sum_exp};
  return ret;
}

template<template<typename, typename, typename> class Epilogue>
Tensor host_softmax_xentropy_backward(
    const at::Tensor &grad_loss,
    const at::Tensor &logits_,
    const at::Tensor &max_log_sum_exp,
    const at::Tensor &labels,
    const float smoothing,
    bool half_to_float) {
  int64_t dim = 1;
  Tensor gI = at::empty_like(logits_);
  if (grad_loss.numel() == 0) {
    return gI;
  }
  auto grad = grad_loss.contiguous();
  static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
  if (grad.dim() == 0) grad = grad.view(1);
  AT_CHECK(dim >=0 && dim < logits_.dim(), "dim must be non-negative and less than input dimensions");
  auto logits = logits_.contiguous();
  if (logits.dim() == 0) logits = logits.view(1);
  int64_t outer_size = 1;
  int64_t dim_size = logits.size(dim);
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= logits.size(i);
  for (int64_t i = dim + 1; i < logits.dim(); ++i)
    inner_size *= logits.size(i);
// See descriptions of kernels above.
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_CHECK(inner_size == 1, "Currently only inner size 1 supported");

  const int ILP = 2;
  dim3 grid(outer_size);
  dim3 block = SoftMax_getBlockSize(ILP, dim_size);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(gI.type(), "host_softmax_xentropy_backward", [&] {
  using accscalar_t = acc_type<scalar_t, true>;
  if (!half_to_float) {
      cunn_SoftMaxXEntropyBackward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
       <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
          gI.data<scalar_t>(), logits.data<scalar_t>(),
          max_log_sum_exp.data<scalar_t>(),
          grad.data<scalar_t>(), labels.data<int64_t>(),
          smoothing, dim_size
  );
  } else {
      cunn_SoftMaxXEntropyBackward<ILP, scalar_t, accscalar_t, accscalar_t, Epilogue>
       <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
          gI.data<scalar_t>(), logits.data<scalar_t>(),
          max_log_sum_exp.data<accscalar_t>(),
          grad.data<accscalar_t>(), labels.data<int64_t>(),
          smoothing, dim_size
  );
  }
  });

  THCudaCheck(cudaGetLastError());
  return gI;
}

std::vector<Tensor> softmax_xentropy_cuda(const Tensor &input, const Tensor &labels, const float smoothing, const bool half_to_float){
  return host_softmax_xentropy<LogSoftMaxForwardEpilogue>(input, labels, smoothing, half_to_float);
}

at::Tensor softmax_xentropy_backward_cuda(
    const at::Tensor &grad_loss,
    const at::Tensor &logits,
    const at::Tensor &max_log_sum_exp,
    const at::Tensor &labels,
    const float smoothing) {
  bool half_to_float = grad_loss.type().scalarType() != logits.type().scalarType();
  if (half_to_float) {
     AT_ASSERTM((grad_loss.type().scalarType() == ScalarType::Float && logits.type().scalarType() == ScalarType::Half), "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
  }
  return host_softmax_xentropy_backward<LogSoftMaxBackwardEpilogue>(grad_loss, logits, max_log_sum_exp, labels, smoothing, half_to_float);
}
