#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/triplet_db_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/data_transformer.hpp"


namespace caffe {

template <typename Dtype>
TripletDBDataLayer<Dtype>::TripletDBDataLayer( const LayerParameter& param )
  : BasePrefetchingDataLayer<Dtype>(param), 
    reader_(param) {
}

template <typename Dtype>
TripletDBDataLayer<Dtype>::~TripletDBDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void TripletDBDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.triplet_data_param().batch_size();
  TripletDatum& triplet_datum = *( reader_.full().peek() );
  Datum datum = UnravelTripletDatumToDatum( triplet_datum, 0 );

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
void TripletDBDataLayer<Dtype>::ShuffleTriplets() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}
*/
template <typename Dtype>
Datum TripletDBDataLayer<Dtype>::UnravelTripletDatumToDatum( const TripletDatum& triplet_datum, int tag ){
  
  CHECK( tag <= 3 ) << "the indicator tag should not be larger than 3 ";
  Datum datum;
  if( tag == 0 )
    datum.set_data( triplet_datum.data_anchor() );
  if( tag == 1 )
    datum.set_data( triplet_datum.data_pos() );
  if( tag == 2 )
    datum.set_data( triplet_datum.data_neg() );

  datum.set_channels( triplet_datum.channels() );
  datum.set_height( triplet_datum.height() );
  datum.set_width( triplet_datum.width() );

  datum.set_encoded( triplet_datum.encoded() );
  
  CHECK( sizeof(datum) > 0 ) << "the datum is not successfully initialized from triplet_datum since it is empty";

  return datum;
}

// This function is called on prefetch thread
template <typename Dtype>
void TripletDBDataLayer<Dtype>::load_batch( Batch<Dtype>* batch ) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

  CHECK( batch->data_.count() );
  CHECK( this->transformed_data_.count() );

  const int batch_size = this->layer_param_.triplet_data_param().batch_size();
  TripletDatum& triplet_datum = *( reader_.full().peek() );
  Datum datum = UnravelTripletDatumToDatum( triplet_datum, 0 );
  //datum.set_data( triplet_datum.data_anchor() );
  //datum.set_encoded( triplet_datum.encoded() );
  //datum.set_channels( triplet_datum.channels() );
  //datum.set_height( triplet_datum.height() );
  //datum.set_width( triplet_datum.width() );

  CHECK( sizeof( datum ) > 0 ) << "the constructed datum is void ";
 
  vector<int> top_shape = this->data_transformer_->InferBlobShape( datum );
  this->transformed_data_.Reshape( top_shape );
  top_shape[0] = 3*batch_size;
  batch->data_.Reshape( top_shape );

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();

  // datum scales
  // const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    //LOG( INFO ) << "item_id = " << item_id;
    timer.Start();
    TripletDatum& triplet_datum = *( reader_.full().pop("Waiting for triplet data") );
    read_time += timer.MicroSeconds();
    timer.Start();

    for (int tri_id=0; tri_id<3; ++tri_id) {
      // get a blob
      //timer.Start();
      Datum datum_query = UnravelTripletDatumToDatum( triplet_datum, tri_id );
      //Datum datum_pos =  UnravelTripletDatumToDatum( triplet_datum, 1 );
      //Datum datum_neg = UnravelTripletDatumToDatum( triplet_datum, 2 );

      int offset = batch->data_.offset( item_id + tri_id * batch_size );
      this->transformed_data_.set_cpu_data( prefetch_data + offset );
      this->data_transformer_->Transform( datum_query, &(this->transformed_data_) );
      //trans_time += timer.MicroSeconds();
    }
    trans_time += timer.MicroSeconds();
    reader_.free().push(const_cast<TripletDatum*>(&triplet_datum));
    //LOG( INFO ) << "Read the " << item_id << "-th batch_size";
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(TripletDBDataLayer);
REGISTER_LAYER_CLASS(TripletDBData);

}  // namespace caffe
#endif  // USE_OPENCV
