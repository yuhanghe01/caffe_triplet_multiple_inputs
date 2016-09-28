#ifndef CAFFE_TRIPLET_DB_DATA_LAYER_HPP_
#define CAFFE_TRIPLET_DB_DATA_LAYER_HPP_
/*
 * Author: Yuhang He
 * Email: heyuhang@dress-plus.com
 * Date: Aug. 11, 2016
 * Note: this header script is implemented for loading triplet leveldb dataset;
 *
 */

#include <string>
#include <utility>
#include <vector>

#include "caffe/triplet_data_reader.hpp"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Provides triplet data to the Net from DB files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class TripletDBDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit TripletDBDataLayer(const LayerParameter& param);
  virtual ~TripletDBDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "TripletDBData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  //we need this function to extract a Datum from a TripletDatum so as to predict the output blob
  //shape.
  /*
 * @param: tag = 0 means extract the anchor image;
 *         tag = 1 means extract the positive image;
 *         tag = 2 means extract the negative image;
 */
 Datum UnravelTripletDatumToDatum( const TripletDatum& triplet_datum, int tag );
 protected:
  //shared_ptr<Caffe::RNG> prefetch_rng_;
  //virtual void ShuffleTriplets();
  virtual void load_batch(Batch<Dtype>* batch);

  TripletDataReader reader_;

  //vector<vector<std::string> > lines_;
  //int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_TRIPLET_DB_DATA_LAYER_HPP_


