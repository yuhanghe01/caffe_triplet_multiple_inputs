#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/triplet_multiple_db_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/data_transformer.hpp"


namespace caffe {

template <typename Dtype>
TripletMultipleDBDataLayer<Dtype>::TripletMultipleDBDataLayer( const LayerParameter& param )
  : BasePrefetchingDataLayer<Dtype>(param), 
    reader_(param) {
}

template <typename Dtype>
TripletMultipleDBDataLayer<Dtype>::~TripletMultipleDBDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void TripletMultipleDBDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.triplet_data_param().batch_size();
  TripletMultipleDatum& triplet_multiple_datum = *( reader_.full().peek() );
  Datum datum = PeepUnravelTripletMultipleDatumToDatum( triplet_multiple_datum, 0 );

  /*
  datum.set_data( triplet_datum.data_anchor() );
  datum.set_encoded( triplet_datum.encoded() );
  datum.set_channels( triplet_datum.channels() );
  datum.set_height( triplet_datum.height() );
  datum.set_width( triplet_datum.width() );
  */
 
  CHECK( sizeof(datum) > 0 ) << "the constructed datum from triplet_datum is void!"; 
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
 
  //calculating the basic top_shape[0] size;
   
  top_shape[0] = 3*batch_size;
  top[0]->Reshape( top_shape );

  if( top_shape.size() == 4 ){
    LOG( INFO ) << "shape string in setup is " << top_shape[0] 
              << " " << top_shape[1] 
              << " " << top_shape[2] 
              << " " << top_shape[3];
  }
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
     << top[0]->width();
}

/*
template <typename Dtype>
void TripletMultipleDBDataLayer<Dtype>::ShuffleTriplets() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}
*/
template <typename Dtype>
int TripletMultipleDBDataLayer<Dtype>::CompImgNumPerBatch( const TripletMultipleDatum& triplet_multiple_datum ){
  CHECK( sizeof(triplet_multiple_datum) > 0 ) << "The input triplet_multiple_datum size should be larger than 0";
  return triplet_multiple_datum.data_anchor_size() + triplet_multiple_datum.data_pos_size() + triplet_multiple_datum.data_neg_size();
}

template <typename Dtype>
std::vector<Datum> TripletMultipleDBDataLayer<Dtype>::UnravelTripletMultipleDatumToDatum( const TripletMultipleDatum& triplet_multiple_datum, int tag ){
  CHECK( tag <= 3 ) << "the indicator tag should be larger than 3 ";
  std::vector< Datum > datum_vec;

  //extract anchor points;
  if( tag == 0 ){
    for( int i = 0; i < triplet_multiple_datum.data_anchor_size(); ++i ){
      Datum datum_tmp;
      //datum_tmp.set_data( triplet_multiple_datum.data_anchor()[i] );
      datum_tmp.set_data( triplet_multiple_datum.data_anchor(0) );
      datum_tmp.set_channels( triplet_multiple_datum.channels() );
      datum_tmp.set_height( triplet_multiple_datum.height() );
      datum_tmp.set_width( triplet_multiple_datum.width() );
      datum_tmp.set_encoded( triplet_multiple_datum.encoded() );

      datum_vec.push_back( datum_tmp );
    }
  }

  if( tag == 1 ){
    for( int i = 0; i < triplet_multiple_datum.data_pos_size(); ++i ){
      Datum datum_tmp;
      //datum_tmp.set_data( triplet_multiple_datum.data_pos()[i] );
      datum_tmp.set_data( triplet_multiple_datum.data_pos(0) );
      datum_tmp.set_channels( triplet_multiple_datum.channels() );
      datum_tmp.set_height( triplet_multiple_datum.height() );
      datum_tmp.set_width( triplet_multiple_datum.width() );
      datum_tmp.set_encoded( triplet_multiple_datum.encoded() );

      datum_vec.push_back( datum_tmp );
    }
  }

  if( tag == 2 ){
    for( int i = 0; i < triplet_multiple_datum.data_neg_size(); ++i ){
      Datum datum_tmp;
      //datum_tmp.set_data( triplet_multiple_datum.data_neg()[i] );
      datum_tmp.set_data( triplet_multiple_datum.data_neg(i) );
      datum_tmp.set_channels( triplet_multiple_datum.channels() );
      datum_tmp.set_height( triplet_multiple_datum.height() );
      datum_tmp.set_width( triplet_multiple_datum.width() );
      datum_tmp.set_encoded( triplet_multiple_datum.encoded() );

      datum_vec.push_back( datum_tmp );
    }
  }
 
  CHECK( datum_vec.size() > 0 ) << "the unravelled datum_vec should be larger than 0"; 
  return datum_vec;
}

template <typename Dtype>
Datum TripletMultipleDBDataLayer<Dtype>::PeepUnravelTripletMultipleDatumToDatum( const TripletMultipleDatum& triplet_multiple_datum, int tag ){
  
  CHECK( tag <= 3 ) << "the indicator tag should not be larger than 3 ";
  Datum datum;
  if( tag == 0 )
    datum.set_data( triplet_multiple_datum.data_anchor(0) );
  if( tag == 1 )
    datum.set_data( triplet_multiple_datum.data_pos(0) );
  if( tag == 2 )
    datum.set_data( triplet_multiple_datum.data_neg(0) );

  datum.set_channels( triplet_multiple_datum.channels() );
  datum.set_height( triplet_multiple_datum.height() );
  datum.set_width( triplet_multiple_datum.width() );

  datum.set_encoded( triplet_multiple_datum.encoded() );
  
  CHECK( sizeof(datum) > 0 ) << "the datum is not successfully initialized from triplet_datum since it is empty";

  return datum;
}

// This function is called on prefetch thread
template <typename Dtype>
void TripletMultipleDBDataLayer<Dtype>::load_batch( Batch<Dtype>* batch ) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

  CHECK( batch->data_.count() );
  CHECK( this->transformed_data_.count() );

  const int batch_size = this->layer_param_.triplet_data_param().batch_size();
  TripletMultipleDatum& triplet_multiple_datum = *( reader_.full().peek() );
  Datum datum = PeepUnravelTripletMultipleDatumToDatum( triplet_multiple_datum, 0 );
  //datum.set_data( triplet_datum.data_anchor() );
  //datum.set_encoded( triplet_datum.encoded() );
  //datum.set_channels( triplet_datum.channels() );
  //datum.set_height( triplet_datum.height() );
  //datum.set_width( triplet_datum.width() );

  CHECK( sizeof( datum ) > 0 ) << "the constructed datum is void ";
 
  vector<int> top_shape = this->data_transformer_->InferBlobShape( datum );
  this->transformed_data_.Reshape( top_shape );
  int img_num_per_batch = CompImgNumPerBatch( triplet_multiple_datum );
  top_shape[0] = img_num_per_batch*batch_size;
  batch->data_.Reshape( top_shape );

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();

  // datum scales
  // const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    //LOG( INFO ) << "item_id = " << item_id;
    timer.Start();
    TripletMultipleDatum& triplet_multiple_datum = *( reader_.full().pop("Waiting for triplet data") );
    read_time += timer.MicroSeconds();
    timer.Start();

    int skip_step = 0;
    int skip_step_prev = 0;
    for (int tri_id=0; tri_id<3; ++tri_id) {
      // get a blob
      //timer.Start();
      std::vector< Datum > datum_query = UnravelTripletMultipleDatumToDatum( triplet_multiple_datum, tri_id );
      //Datum datum_pos =  UnravelTripletDatumToDatum( triplet_datum, 1 );
      //Datum datum_neg = UnravelTripletDatumToDatum( triplet_datum, 2 );

      //skip_step_prev += datum_query.size();
      if( tri_id > 0 ){
        skip_step = skip_step_prev;
      }

      skip_step_prev += datum_query.size();
      int skip_current = datum_query.size();
      for( int i = 0; i < datum_query.size(); ++i ){
        int offset_tmp = batch->data_.offset( item_id*skip_current + batch_size * skip_step  + i );
        this->transformed_data_.set_cpu_data( prefetch_data + offset_tmp );
        this->data_transformer_->Transform( datum_query, &(this->transformed_data_) );
      }
      //int offset = batch->data_.offset( item_id + tri_id * batch_size );
      //this->transformed_data_.set_cpu_data( prefetch_data + offset );
      //this->data_transformer_->Transform( datum_query, &(this->transformed_data_) );
      //trans_time += timer.MicroSeconds();
    }
    trans_time += timer.MicroSeconds();
    reader_.free().push(const_cast<TripletMultipleDatum*>(&triplet_multiple_datum));
    //LOG( INFO ) << "Read the " << item_id << "-th batch_size";
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(TripletMultipleDBDataLayer);
REGISTER_LAYER_CLASS(TripletMultipleDBData);

}  // namespace caffe
#endif  // USE_OPENCV
