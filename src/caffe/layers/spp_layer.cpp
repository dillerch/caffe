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
	LayerParameter SPPLayer<Dtype>::CreatePoolingParam(const int num_bins, const int bottom_h, const int bottom_w, const SPPParameter spp_param) {
		LayerParameter pooling_param;

		//Calculate the kernel(window) size and the stride as denoted in the paper for both height and width
		int kernel_h = ceil(bottom_h / static_cast<double>(num_bins));
		int stride_h = floor(bottom_h / static_cast<double>(num_bins));

		int kernel_w = ceil(bottom_w / static_cast<double>(num_bins));
		int stride_w = floor(bottom_w / static_cast<double>(num_bins));

		//Set the parameters
		pooling_param.mutable_pooling_param()->set_kernel_h(kernel_h);
	 	pooling_param.mutable_pooling_param()->set_kernel_w(kernel_w);
	 	pooling_param.mutable_pooling_param()->set_stride_h(stride_h);
		pooling_param.mutable_pooling_param()->set_stride_w(stride_w);

		//Set the pooling method
		switch (spp_param.pool()) {
		case SPPParameter_PoolMethod_MAX:
			pooling_param.mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_MAX);
			break;
		case SPPParameter_PoolMethod_AVE:
			pooling_param.mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_AVE);
			break;
		case SPPParameter_PoolMethod_STOCHASTIC:
			pooling_param.mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_STOCHASTIC);
			break;
		default:
			LOG(FATAL) << "Unknown pooling method.";
		}

		return pooling_param;
	}

	template <typename Dtype>
	void SPPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const SPPParameter spp_param = this->layer_param_.spp_param();

		//Store parameters as reference values for later reshapes
		num_ = bottom[0]->num();
		channels_ = bottom[0]->channels();
		bottom_h_ = bottom[0]->height();
		bottom_w_ = bottom[0]->width();
		reshaped_first_time_ = false;

		//Check for valid input dimensions
		CHECK_GT(bottom_h_, 0) << "Input dimensions cannot be zero.";
		CHECK_GT(bottom_w_, 0) << "Input dimensions cannot be zero.";

		//Support the pyramid type {1,2,3,4} and {1,2,3,6} as used in the paper or 2^i for i = 0..pyramid_height
		switch (spp_param.pyramid_type()) {
		case SPPParameter_PyramidType_SPP_NET4:
			pyramid_bins_.push_back(1); pyramid_bins_.push_back(2); pyramid_bins_.push_back(3); pyramid_bins_.push_back(4);
			pyramid_height_ = 4;
			custom_pyramid_ = false;
			break;
		case SPPParameter_PyramidType_SPP_NET6:
			pyramid_bins_.push_back(1); pyramid_bins_.push_back(2); pyramid_bins_.push_back(3); pyramid_bins_.push_back(6);
			pyramid_height_ = 4;
			custom_pyramid_ = false;
			break;
		case SPPParameter_PyramidType_CUSTOM:
			pyramid_height_ = spp_param.pyramid_height();
			custom_pyramid_ = true;
			break;
		default:
			LOG(FATAL) << "Unknown pyramid type.";
		}
		CHECK_GT(pyramid_height_, 1) << "Pyramid height must be at least 2.";

		//Clear all vectors
		split_top_vec_.clear();
		pooling_bottom_vecs_.clear();
		pooling_layers_.clear();
		pooling_top_vecs_.clear();
		flatten_layers_.clear();
		flatten_top_vecs_.clear();
		concat_bottom_vec_.clear();

		// split layer output holders setup
		for (int i = 0; i < pyramid_height_; i++)
			split_top_vec_.push_back(new Blob<Dtype>());

		//Setup the split layer which will split the input from bottom to split_top_vec_
		LayerParameter split_param;
		split_layer_.reset(new SplitLayer<Dtype>(split_param));
		split_layer_->SetUp(bottom, split_top_vec_);

		for (int i = 0; i < pyramid_height_; i++) {
			//If this is a custom pyramid, use 2^i as number of bins; if not, use pre-defined number of bins
			const int num_bins = custom_pyramid_ ? pow(2, i) : pyramid_bins_[i];

			//# POOLING LAYERS
			//Create the pooling parameter with number of bins, bottom dimensions and the spp parameter
			const LayerParameter pooling_param = CreatePoolingParam(num_bins, bottom_h_, bottom_w_, spp_param);

			//Fill the pooling bottom vectors with split layer top vectors
			pooling_bottom_vecs_.push_back(new vector<Blob<Dtype>*>);
			pooling_bottom_vecs_[i]->push_back(split_top_vec_[i]);

			//Fill the pooling top vectors into pooling_top_vecs_
			pooling_top_vecs_.push_back(new vector<Blob<Dtype>*>);
			pooling_top_vecs_[i]->push_back(new Blob<Dtype>());

			//Setup pooling layers
			pooling_layers_.push_back(shared_ptr<PoolingLayer<Dtype> > (new PoolingLayer<Dtype>(pooling_param)));
			pooling_layers_[i]->SetUp(*pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);

			//# FLATTEN LAYERS
			//Create single flatten output layer and multiple flatten top vectors
			flatten_top_vecs_.push_back(new vector<Blob<Dtype>*>);
			flatten_top_vecs_[i]->push_back(new Blob<Dtype>());

			//Setup flatten layers
			LayerParameter flatten_param;
			flatten_layers_.push_back(new FlattenLayer<Dtype>(flatten_param));
			flatten_layers_[i]->SetUp(*pooling_top_vecs_[i], *flatten_top_vecs_[i]);

			//Current flatten top output is a bottom input to concat layer
			concat_bottom_vec_.push_back((*flatten_top_vecs_[i])[0]);
		}

		//Setup concat layer
		LayerParameter concat_param;
		concat_layer_.reset(new ConcatLayer<Dtype>(concat_param));
		concat_layer_->SetUp(concat_bottom_vec_, top);
	}

	template <typename Dtype>
	void SPPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, corresponding to (num, channels, height, width)";
		// Do nothing if bottom shape is unchanged since last Reshape
		if (num_ == bottom[0]->num() && channels_ == bottom[0]->channels() &&
				bottom_h_ == bottom[0]->height() && bottom_w_ == bottom[0]->width() &&
				reshaped_first_time_) return;

		num_ = bottom[0]->num();
		channels_ = bottom[0]->channels();
		bottom_h_ = bottom[0]->height();
		bottom_w_ = bottom[0]->width();
		reshaped_first_time_ = true;

		SPPParameter spp_param = this->layer_param_.spp_param();
		split_layer_->Reshape(bottom, split_top_vec_);

		for (int i = 0; i < pyramid_height_; i++) {
			int num_bins = custom_pyramid_ ? pow(2, i) : pyramid_bins_[i];
			LayerParameter pooling_param = CreatePoolingParam(num_bins, bottom_h_, bottom_w_, spp_param);

			pooling_layers_[i].reset(new PoolingLayer<Dtype>(pooling_param));
			pooling_layers_[i]->SetUp(*pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);
			pooling_layers_[i]->Reshape(*pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);
			flatten_layers_[i]->Reshape(*pooling_top_vecs_[i], *flatten_top_vecs_[i]);
		}
		concat_layer_->Reshape(concat_bottom_vec_, top);
	}

	template <typename Dtype>
	void SPPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//Split
		split_layer_->Forward(bottom, split_top_vec_);
		//Pool and flatten
		for (int i = 0; i < pyramid_height_; ++i) {
			pooling_layers_[i]->Forward(*pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);
			flatten_layers_[i]->Forward(*pooling_top_vecs_[i], *flatten_top_vecs_[i]);
		}
		//Concat
		concat_layer_->Forward(concat_bottom_vec_, top);
	}

	template <typename Dtype>
	void SPPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) return;
		vector<bool> concat_propagate_down(pyramid_height_, true);
		concat_layer_->Backward(top, concat_propagate_down, concat_bottom_vec_);
		for (int i = 0; i < pyramid_height_; ++i) {
			flatten_layers_[i]->Backward(*flatten_top_vecs_[i], propagate_down, *pooling_top_vecs_[i]);
			pooling_layers_[i]->Backward(*pooling_top_vecs_[i], propagate_down, *pooling_bottom_vecs_[i]);
		}

		split_layer_->Backward(split_top_vec_, propagate_down, bottom);
	}


	INSTANTIATE_CLASS(SPPLayer);
	REGISTER_LAYER_CLASS(SPP);

}
