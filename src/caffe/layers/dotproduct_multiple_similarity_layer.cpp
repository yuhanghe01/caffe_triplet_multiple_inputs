#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dotproduct_multiple_similarity_layer.hpp"

namespace caffe {

template <typename Dtype>
void DotProductMultipleSimilarityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //bottom[0] is the anchor point
  //bottom[1] is the positive/negative point
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  Dtype * sim = top[0]->mutable_cpu_data();
  const Dtype * pa = bottom[0]->cpu_data();
  const Dtype * pb = bottom[1]->cpu_data();
  int num_pos_neg = bottom[1]->num();
  CHECK( num_pos_neg > num ) << "num should be smaller than num_pos_neg";
  CHECK( num_pos_neg%num == 0 ) << "the num_pos_neg should be devisible of num";
  //num is the batch_size;
  //dim is the dimension length;
  for (int i=0; i<num; ++i) {
    Dtype sim_tmp = (Dtype)0.;
    for( int j = 0; j < num_pos_neg/num; ++j ){
      sim_tmp += caffe_cpu_dot( dim, pa, pb );
      pb += dim;
    }
    sim[i] = sim_tmp/(num_pos_neg/num);
    //sim[i] = caffe_cpu_dot(dim, pa, pb);
    //pa += dim; pb += dim;
    pa += dim;
    //pb += dim;
  }
}

template <typename Dtype>
void DotProductMultipleSimilarityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  int num_pos_neg = bottom[1]->num();
  int num_pos_neg_per_batch = num_pos_neg/num;

  const Dtype * top_diff = top[0]->cpu_diff();
  for (int i=0; i<num; ++i) {
    //update the pos/neg diff;
    for( int j = 0; j < num_pos_neg_per_batch; ++j ){
      caffe_cpu_scale( dim, top_diff[i]/num_pos_neg_per_batch,
      bottom[0]->cpu_data() + bottom[0]->offset(i),
      bottom[1]->mutable_cpu_diff() + bottom[1]->offset(i*num_pos_neg_per_batch + j) );
    }
    //caffe_cpu_scale(dim, top_diff[i],
    //    bottom[0]->cpu_data() + bottom[0]->offset(i),
    //    bottom[1]->mutable_cpu_diff() + bottom[1]->offset(i));

    //update the anchor's diff. We assume that anchor point-i in bottom[0] corresponds to 
    //pos/neg point i*(num_pos_neg_per_batch), i*(num_pos_neg_per_batch) + 1, ... Note that
    //i starts with 0
    std::vector< Dtype* > pos_neg_diff_total;
    for( int j = 0; j < num_pos_neg_per_batch; ++j ){
      Dtype* diff_tmp = new Dtype[ dim ];
      caffe_cpu_scale( dim, top_diff[i],
          bottom[1]->cpu_data() + bottom[1]->offset( i*num_pos_neg_per_batch + j ),
          diff_tmp);
      pos_neg_diff_total.push_back( diff_tmp );
      delete [] diff_tmp;
    }
    Dtype* anchor_diff = new Dtype[dim];
    for( int j = 0; j < dim; ++j ){
       *(anchor_diff + j) = Dtype( 0 );
    }
    for( int j = 0; j < pos_neg_diff_total.size(); ++j ){
      for( int k = 0; k < dim; ++k ){
        *(anchor_diff + k ) += *(pos_neg_diff_total[j] + k);
      }
    }
    for( int j = 0; j < dim; ++j ){
      *(anchor_diff + j) = *(anchor_diff + j)/pos_neg_diff_total.size();
    }
    //copy anchor_diff data to bottom[0]->mutable_cpu_diff() + bottom[0]->offset(i);
    caffe_cpu_scale( dim, Dtype(1), anchor_diff, bottom[0]->mutable_cpu_diff() + 
                     bottom[1] -> offset(i) );
    //caffe_cpu_scale(dim, top_diff[i],
    //    bottom[1]->cpu_data() + bottom[1]->offset(i),
    //    bottom[0]->mutable_cpu_diff() + bottom[0]->offset(i));
    if( anchor_diff != NULL ){
      delete [] anchor_diff;
    }
    pos_neg_diff_total.clear();
  }
}

#ifdef CPU_ONLY
STUB_GPU(DotProductMultipleSimilarityLayer);
#endif

INSTANTIATE_CLASS(DotProductMultipleSimilarityLayer);
REGISTER_LAYER_CLASS(DotProductMultipleSimilarity);
}  // namespace caffe
