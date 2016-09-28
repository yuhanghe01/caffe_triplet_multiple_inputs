#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/naive_triplet_multiple_loss_layer.hpp"

/*
 *Author: Yuhang He
 *Email: yuhanghe@whu.edu.cn
 *Date: Aug. 28, 2016
 *Note: this script is implemented for triplet loss loss with multiple positive and
 *negative inputs.
 */
namespace caffe {

template <typename Dtype>
void NaiveTripletMultipleLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // accuracy do not contribute to the loss
  this->layer_param_.add_loss_weight(Dtype(0));

  int num = bottom[0]->num();
  CHECK(num > 0) << "Number of images must be positive.";
  //CHECK(num % 3 == 0) << "Number of images must be multiple of 3.";
  //int batch_size = num / 3;
  batch_size_ = this->layer_param_.multiple_triplet_loss_param().batch_size();
  anchor_len_ = this->layer_param_.multiple_triplet_loss_param().anchor_len();
  pos_len_ = this->layer_param_.multiple_triplet_loss_param().pos_len();
  neg_len_ = this->layer_param_.multiple_triplet_loss_param().neg_len();
  //CHECK( num == batch_size_*(anchor_len_ + pos_len_ + neg_len_) ) << "the following equation should be met: num == batch_size_ * (anchor_len_ + pos_len_ + neg_len_)"; 
  LayerParameter split_param;
  split_param.set_type("Slice");
  split_param.mutable_slice_param()->set_axis(0);
  split_param.mutable_slice_param()->add_slice_point( batch_size_*anchor_len_ );
  split_param.mutable_slice_param()->add_slice_point( batch_size_*(anchor_len_ + pos_len_ ));
  split_layer_ = LayerRegistry<Dtype>::CreateLayer(split_param);
  split_bottom_vec_.clear();
  split_bottom_vec_.push_back(bottom[0]);
  split_top_vec_.clear();
  split_top_vec_.push_back(&qry_feat_);
  split_top_vec_.push_back(&pos_feat_);
  split_top_vec_.push_back(&neg_feat_);
  split_layer_->SetUp(split_bottom_vec_, split_top_vec_);

  const string & sim_type = this->layer_param_.multiple_triplet_loss_param().sim_type();

  LayerParameter pos_sim_param;
  pos_sim_param.set_type(sim_type);
  pos_sim_layer_ = LayerRegistry<Dtype>::CreateLayer(pos_sim_param);
  pos_sim_bottom_vec_.clear();
  pos_sim_bottom_vec_.push_back(&qry_feat_);
  pos_sim_bottom_vec_.push_back(&pos_feat_);
  pos_sim_top_vec_.clear();
  pos_sim_top_vec_.push_back(&pos_sim_);
  pos_sim_layer_->SetUp(pos_sim_bottom_vec_, pos_sim_top_vec_);

  LayerParameter neg_sim_param;
  neg_sim_param.set_type(sim_type);
  neg_sim_layer_ = LayerRegistry<Dtype>::CreateLayer(neg_sim_param);
  neg_sim_bottom_vec_.clear();
  neg_sim_bottom_vec_.push_back(&qry_feat_);
  neg_sim_bottom_vec_.push_back(&neg_feat_);
  neg_sim_top_vec_.clear();
  neg_sim_top_vec_.push_back(&neg_sim_);
  neg_sim_layer_->SetUp(neg_sim_bottom_vec_, neg_sim_top_vec_);
}

template <typename Dtype>
void NaiveTripletMultipleLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  top[1]->Reshape(loss_shape);

  split_layer_->Reshape(split_bottom_vec_, split_top_vec_);
  pos_sim_layer_->Reshape(pos_sim_bottom_vec_, pos_sim_top_vec_);
  neg_sim_layer_->Reshape(neg_sim_bottom_vec_, neg_sim_top_vec_);
}

template <typename Dtype>
void NaiveTripletMultipleLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  split_layer_->Forward(split_bottom_vec_, split_top_vec_);
  pos_sim_layer_->Forward(pos_sim_bottom_vec_, pos_sim_top_vec_);
  neg_sim_layer_->Forward(neg_sim_bottom_vec_, neg_sim_top_vec_);

  const Dtype* pos_sim = pos_sim_.cpu_data();
  const Dtype* neg_sim = neg_sim_.cpu_data();
  Dtype* per_triplet_loss = pos_sim_.mutable_cpu_diff();
  int count = pos_sim_.count();

  Dtype loss = 0;
  Dtype accuracy = 0;
  for (int i=0; i<count; ++i) {
    per_triplet_loss[i] = std::max(Dtype(0),
        this->layer_param_.triplet_loss_param().margin()
        - pos_sim[i] + neg_sim[i]);
    loss += per_triplet_loss[i];
    accuracy += (pos_sim[i] > neg_sim[i] ? 1 : 0);
  }
  top[0]->mutable_cpu_data()[0] = loss / count;
  top[1]->mutable_cpu_data()[0] = accuracy / count;
}

template <typename Dtype>
void NaiveTripletMultipleLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    bool sample = this->layer_param_.triplet_loss_param().sample();
    Dtype* pos_diff = pos_sim_.mutable_cpu_diff();
    Dtype* neg_diff = neg_sim_.mutable_cpu_diff();
    const Dtype* pos_sim = pos_sim_.cpu_data();
    const Dtype* neg_sim = neg_sim_.cpu_data();
    int count = pos_sim_.count();
    for (int i=0; i<count; ++i) {
      if (pos_diff[i] && (!sample || pos_sim[i] > neg_sim[i])) {
        pos_diff[i] = -1;
        neg_diff[i] = 1;
      } else {
        pos_diff[i] = 0;
        neg_diff[i] = 0;
      }
    }
    pos_sim_layer_->Backward(pos_sim_top_vec_, propagate_down, pos_sim_bottom_vec_);
    neg_sim_layer_->Backward(neg_sim_top_vec_, propagate_down, neg_sim_bottom_vec_);
    split_layer_->Backward(split_top_vec_, propagate_down, split_bottom_vec_);
  }
}

INSTANTIATE_CLASS(NaiveTripletMultipleLossLayer);
REGISTER_LAYER_CLASS(NaiveTripletMultipleLoss);

}  // namespace caffe
