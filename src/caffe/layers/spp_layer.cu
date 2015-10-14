#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef __CDT_PARSER__
#define __global__
#define CUDA_KERNEL_LOOP(i, n) for (int i = 0; i < (n); ++i)
#endif

namespace caffe {

template<typename Dtype>
__global__ void SPPForward(const int num_threads,
    const Dtype* const bottom_data, Dtype* const top_data, int* const mask,
    const int bottom_width, const int bottom_height,
    const int num_bins_w, const int num_bins_h, const int total_num_bins,
    const Dtype bin_size_w, const Dtype bin_size_h,
    const int channels, const int previous_bins) {
  //Run a CUDA kernel loop; grid stride looping
  CUDA_KERNEL_LOOP(index, num_threads) {
    //Get current position
    const int n = index / num_bins_w / num_bins_h / channels;
    const int c = (index / num_bins_w / num_bins_h) % channels;
    const int nbh = (index / num_bins_w) % num_bins_h;
    const int nbw = index % num_bins_w;

    //Calculate bin start and end coordinates
    const int wstart = max(static_cast<int>(floor(static_cast<Dtype>(nbw) * bin_size_w)), 0);
    const int hstart = max(static_cast<int>(floor(static_cast<Dtype>(nbh) * bin_size_h)), 0);
    const int wend = min(static_cast<int>(ceil(static_cast<Dtype>(nbw + 1) * bin_size_w)), bottom_width);
    const int hend = min(static_cast<int>(ceil(static_cast<Dtype>(nbh + 1) * bin_size_h)), bottom_height);

    //Max val and idx in registers
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;

    //Calculate current pointer position
    const Dtype* const bottom_slice = bottom_data + (n * channels + c) * bottom_height * bottom_width;

    //Max pooling
    for(int h = hstart; h < hend; ++h) {
      for(int w = wstart; w < wend; ++w) {
        const int pos = h * bottom_width + w;
        if(bottom_slice[pos] > maxval) {
          maxval = bottom_slice[pos];
          maxidx = pos;
        }
      }
    }
    //Write results to global memory
    const int bin_index = (n * channels + c) * total_num_bins + previous_bins + nbh * num_bins_w + nbw;
    top_data[bin_index] = maxval;
    mask[bin_index] = maxidx;
  }
}

template<typename Dtype>
void SPPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    //Get top, bottom and mask data
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mask = max_idx_.mutable_gpu_data();

    //Store previous pyramid bins
    int previous_bins = 0;

    //Loop over pyramid layers
    for(int p_layer = 0; p_layer < pyramid_height_; ++p_layer) {
      //Calculate bin width and height
      Dtype bin_size_w = static_cast<Dtype>(bottom_w_) / static_cast<Dtype>(num_bins_w_[p_layer]);
      Dtype bin_size_h = static_cast<Dtype>(bottom_h_) / static_cast<Dtype>(num_bins_h_[p_layer]);

      //The number of workers we will spawn on the GPU instead of looping over num, channels and bins
      const int count = num_ * channels_ * num_bins_w_[p_layer] * num_bins_h_[p_layer];

      //Launch CUDA kernel
      SPPForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count,
        bottom_data, top_data, mask,
        bottom_w_, bottom_h_,
        num_bins_w_[p_layer], num_bins_h_[p_layer], total_num_bins_,
        bin_size_w, bin_size_h,
        channels_, previous_bins);

      //Update previous bins
      previous_bins += num_bins_h_[p_layer] * num_bins_w_[p_layer];
    }

    CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void SPPBackward(const int num_threads,
    Dtype* const bottom_diff, const Dtype* const top_diff, const int* const mask,
    const int bottom_width, const int bottom_height,
    const int num_bins_w, const int num_bins_h, const int total_num_bins,
    const Dtype bin_size_w, const Dtype bin_size_h,
    const int channels, const int previous_bins) {
  //Run a CUDA kernel loop; grid stride looping
  CUDA_KERNEL_LOOP(index, num_threads) {
    //Get current position
    const int n = index / bottom_width / bottom_height / channels;
    const int c = (index / bottom_width / bottom_height) % channels;
    const int h = (index / bottom_width) % bottom_height;
    const int w = index % bottom_width;

    //Calculate bin start and end coordinates
    const int wstart = max(static_cast<int>(floor(static_cast<Dtype>(w) / bin_size_w)), 0);
    const int hstart = max(static_cast<int>(floor(static_cast<Dtype>(h) / bin_size_h)), 0);
    const int wend = min(static_cast<int>(ceil(static_cast<Dtype>(w) / bin_size_w + 1)), num_bins_w);
    const int hend = min(static_cast<int>(ceil(static_cast<Dtype>(h) / bin_size_h + 1)), num_bins_h);

    //Calculate current pointer position
    const int offset = (n * channels + c) * total_num_bins;
    const Dtype* const top_diff_slice = top_diff + offset;
    const int* const mask_slice = mask + offset;

    //Gradient in register
    Dtype gradient = 0;

    //Accumulate gradient
    for (int ph = hstart; ph < hend; ++ph)
      for (int pw = wstart; pw < wend; ++pw)
        if (mask_slice[previous_bins + ph * num_bins_w + pw] == h * bottom_width + w)
          gradient += top_diff_slice[previous_bins + ph * num_bins_w + pw];

    //Write gradient to global memory
    bottom_diff[index] = gradient;
  }
}
	
template<typename Dtype>
void SPPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if(!propagate_down[0]) return;

  //Initialize bottom diff
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_gpu_diff());

  //Get top diff, bottom diff and mask
  const Dtype* const top_diff = top[0]->gpu_diff();
  Dtype* const bottom_diff = bottom[0]->mutable_gpu_diff();
  const int* const mask = max_idx_.gpu_data();

  //Store previous pyramid bins
  int previous_bins = 0;

  for(int p_layer = 0; p_layer < pyramid_height_; ++p_layer) {
    //The number of workers we will spawn on the GPU: One per bottom data point
    const int count = bottom[0]->count();

    //Calculate bin width and height
    Dtype bin_size_w = static_cast<Dtype>(bottom_w_) / static_cast<Dtype>(num_bins_w_[p_layer]);
    Dtype bin_size_h = static_cast<Dtype>(bottom_h_) / static_cast<Dtype>(num_bins_h_[p_layer]);

    //Launch CUDA kernel
    SPPBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count,
        bottom_diff, top_diff, mask,
        bottom_w_, bottom_h_,
        num_bins_w_[p_layer], num_bins_h_[p_layer], total_num_bins_,
        bin_size_w, bin_size_h,
        channels_, previous_bins);

    //Update previous bins
    previous_bins += num_bins_h_[p_layer] * num_bins_w_[p_layer];
  }
  CUDA_POST_KERNEL_CHECK;
}
	
	INSTANTIATE_LAYER_GPU_FUNCS(SPPLayer);
}

