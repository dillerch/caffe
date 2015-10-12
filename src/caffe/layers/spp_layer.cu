/*#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef __CDT_PARSER__
#define __global__
#define CUDA_KERNEL_LOOP(a, b)
#endif

namespace caffe {

template<typename Dtype>
__global__
void SPPForward(const int num_threads,
  const Dtype* const bottom_data, Dtype* const top_data, int* mask,
  const int bottom_width, const int bottom_height,
  const int num_bins_w, const int num_bins_h,
  const Dtype bin_size_w, const Dtype bin_size_h,
  const int channels, const int previous_bins) {
    //Run a CUDA kernel loop
    CUDA_KERNEL_LOOP(index, num_threads) {
      //Get current position
      const int n = index / num_bins_w / num_bins_h / channels;
      const int c = (index / num_bins_w / num_bins_h) % channels;
      const int nbh = (index / num_bins_w) % num_bins_h;
      const int nbw = index % num_bins_w;

      //Calculate start and end
      int hstart = max(static_cast<int>(floor(static_cast<Dtype>(nbh) * bin_size_h)), 0);
      int wstart = max(static_cast<int>(floor(static_cast<Dtype>(nbw) * bin_size_w)), 0);
      int hend = min(static_cast<int>(ceil(static_cast<Dtype>(nbh + 1) * bin_size_h)), bottom_height);
      int wend = min(static_cast<int>(ceil(static_cast<Dtype>(nbw + 1) * bin_size_w)), bottom_width);

      Dtype maxval = -FLT_MAX;
      int maxidx = -1;
      const Dtype* const bottom_slice = bottom_data + (n * channels + c) * bottom_width * bottom_height;
      for(int h = hstart; h < hend; ++h) {
        for(int w = wstart; w < wend; ++w) {
          if(bottom_slice[h * bottom_width + w] > maxval) {
            maxidx = h * bottom_width + w;
            maxval = bottom_slice[maxidx];
          }
        }
      }
      top_data[previous_bins + index] = maxval;
      mask[previous_bins + index] = maxidx;
  }
}

template <typename Dtype>
void SPPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    //Get top, bottom and mask data
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mask = max_idx_.mutable_gpu_data();

    int previous_bins = 0;
    //Loop over pyramid layers
    for(int i = 0; i < pyramid_height_; ++i) {
      const int count = num_bins_w_[i] * num_bins_h_[i] * num_ * channels_;
      Dtype bin_size_w = static_cast<Dtype>(bottom_w_) / static_cast<Dtype>(num_bins_w_[p_layer]);
      Dtype bin_size_h = static_cast<Dtype>(bottom_h_) / static_cast<Dtype>(num_bins_h_[p_layer]);

      SPPForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count,
        bottom_data, top_data, mask,
        bottom_w_, bottom_h_,
        num_bins_w_[i], num_bins_h_[i],
        bin_size_w, bin_size_h,
        channels_, previous_bins);

      previous_bins += num_bins_h_[p_layer] * num_bins_w_[p_layer];
    }
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__
void SPPBackward(const int num_threads,
    const Dtype* const bottom_diff, Dtype* const top_diff, int* mask,
    const int bottom_width, const int bottom_height,
    const int num_bins_w, const int num_bins_h,
    const Dtype bin_size_w, const Dtype bin_size_h,
    const int channels, const int previous_bins) {
  //Run a CUDA kernel loop
  CUDA_KERNEL_LOOP(index, num_threads) {
    //Get current position
    const int n = index / bottom_width / bottom_height / channels;
    const int c = (index / bottom_width / bottom_height) % channels;
    const int h = (index / bottom_width) % bottom_height;
    const int w = index % bottom_width;

    const int phstart = (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart = (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * bin_size_h * bin_size_w;
    const Dtype* const top_diff_slice = top_diff + offset;
    const int* const mask_slice = mask + offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_slice[ph * pooled_width + pw] == h * width + w) {
          gradient += top_diff_slice[ph * pooled_width + pw];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}
	
template <typename Dtype>
void SPPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) return;
  //Get top and bottom diff
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* mask = max_idx_.gpu_data();

  int previous_bins = 0;
  for(int i = 0; i < pyramid_height_; ++i) {
    SPPBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count,
        bottom_diff, top_diff, mask,
        bottom_w_, bottom_h_,
        num_bins_w_[i], num_bins_h_[i],
        bin_size_w, bin_size_h,
        channels_, previous_bins);
    previous_bins += num_bins_h_[p_layer] * num_bins_w_[p_layer];
  }
  CUDA_POST_KERNEL_CHECK;
}
	
	INSTANTIATE_LAYER_GPU_FUNCS(SPPLayer);
}*/
