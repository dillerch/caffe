#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	using std::min;
	using std::max;

	template <typename Dtype>
	void SPPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//Get SPP Parameter
		const SPPParameter spp_param = this->layer_param_.spp_param();

    //Store input dimensions
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    bottom_h_ = bottom[0]->height();
    bottom_w_ = bottom[0]->width();

		//Check for valid input dimensions
		CHECK_GT(bottom_h_, 0) << "Input dimensions cannot be zero.";
		CHECK_GT(bottom_w_, 0) << "Input dimensions cannot be zero.";

		//Check for valid input parameters
		CHECK_GT(spp_param.pyramid_shape().size(), 0) << "Spatial Pyramid must have at least one layer.";

		//Set pyramid layers (num_bins per layer) and total number of bins
		total_num_bins_ = 0;
		pyramid_height_ = spp_param.pyramid_shape().size();
		num_bins_w_.resize(pyramid_height_); num_bins_h_.resize(pyramid_height_);
		LOG(INFO) << "Pyramid shape is";
		for(int p_layer = 0; p_layer < pyramid_height_; ++p_layer) {
      CHECK_GT(spp_param.pyramid_shape().Get(p_layer), 0) << "number of bins in pyramid shape must be > 0";
			num_bins_w_[p_layer] = num_bins_h_[p_layer] = spp_param.pyramid_shape().Get(p_layer);
			total_num_bins_ += num_bins_w_[p_layer] * num_bins_h_[p_layer];
			LOG(INFO) << num_bins_w_[p_layer] << "x" << num_bins_h_[p_layer];
		}
		LOG(INFO) << "Total number of bins is " << total_num_bins_;
	}

	template <typename Dtype>
	void SPPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
		      << "corresponding to (num, channels, height, width)";

		//Change input dimensions
		channels_ = bottom[0]->channels();
		bottom_h_ = bottom[0]->height();
		bottom_w_ = bottom[0]->width();

    //Top shape passes bottom num and channels and a third dimension, the total number of bins
    vector<int> top_shape;
    top_shape.push_back(num_);
    top_shape.push_back(channels_);
    top_shape.push_back(total_num_bins_);

		//Reshape max index blob
		max_idx_.Reshape(top_shape);

		//Reshape top
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void SPPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	  //Store previous pyramid bins
	  int previous_bins = 0;

		//Loop over pyramid layers
		for(int p_layer = 0; p_layer < pyramid_height_; ++p_layer) {
	    //Get top, bottom and mask data
	    const Dtype* bottom_data = bottom[0]->cpu_data();
	    Dtype* top_data = top[0]->mutable_cpu_data();
	    int* mask = max_idx_.mutable_cpu_data();
	    //Calculate bin width and height
			Dtype bin_size_w = static_cast<Dtype>(bottom_w_) / static_cast<Dtype>(num_bins_w_[p_layer]);
			Dtype bin_size_h = static_cast<Dtype>(bottom_h_) / static_cast<Dtype>(num_bins_h_[p_layer]);
			//Reset top data and mask
      caffe_set(top[0]->count(), -1, mask);
      caffe_set(top[0]->count(), Dtype(-FLT_MAX), top_data);
			//Loop over num, channels, bin h and w
			for (int n = 0; n < num_; ++n) {
				for (int c = 0; c < channels_; ++c) {
					for (int nbh = 0; nbh < num_bins_h_[p_layer]; ++nbh) {
						for (int nbw = 0; nbw < num_bins_w_[p_layer]; ++nbw) {
						  //Calculate bin start and end coordinates
              const int wstart = max(static_cast<int>(floor(static_cast<Dtype>(nbw) * bin_size_w)), 0);
							const int hstart = max(static_cast<int>(floor(static_cast<Dtype>(nbh) * bin_size_h)), 0);
              const int wend = min(static_cast<int>(ceil(static_cast<Dtype>(nbw + 1) * bin_size_w)), bottom_w_);
							const int hend = min(static_cast<int>(ceil(static_cast<Dtype>(nbh + 1) * bin_size_h)), bottom_h_);
							const int bin_index = previous_bins + nbh * num_bins_w_[p_layer] + nbw;
							//Max pooling within one bin
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									const int index = h * bottom_w_ + w;
									if (bottom_data[index] > top_data[bin_index]) {
										top_data[bin_index] = bottom_data[index];
										mask[bin_index] = index;
									}
								}
							}
						}
					}
					//Shift pointers
          bottom_data += bottom[0]->offset(0, 1);
          top_data += top[0]->offset(0, 1);
          mask += top[0]->offset(0, 1);
				}
			}
			previous_bins += num_bins_h_[p_layer] * num_bins_w_[p_layer];
		}
	}

	template <typename Dtype>
	void SPPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) return;

		//Memset bottom diffs to 0
    caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());

    int previous_bins = 0;
    //Loop over pyramid layers
    for(int p_layer = 0; p_layer < pyramid_height_; ++p_layer) {
      //Get top and bottom diffs
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const int* mask = max_idx_.cpu_data();
      //Loop over num, channels, bin h and w
      for (int n = 0; n < top[0]->num(); ++n) {
        for (int c = 0; c < channels_; ++c) {
          for (int nbh = 0; nbh < num_bins_h_[p_layer]; ++nbh) {
            for (int nbw = 0; nbw < num_bins_w_[p_layer]; ++nbw) {
              const int index = previous_bins + nbh * num_bins_w_[p_layer] + nbw;
              const int bottom_index = mask[index];
              bottom_diff[bottom_index] += top_diff[index];
            }
          }
          //Shift pointers
          bottom_diff += bottom[0]->offset(0, 1);
          top_diff += top[0]->offset(0, 1);
          mask += top[0]->offset(0, 1);
        }
      }
      previous_bins += num_bins_h_[p_layer] * num_bins_w_[p_layer];
    }
  }

	INSTANTIATE_CLASS(SPPLayer);
	REGISTER_LAYER_CLASS(SPP);

}
