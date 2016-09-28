/*
 *@Author: Yuhang He
 *@Date: Mar. 28, 2016
 *@Email: yuhanghe@whu.edu.cn
 */
#include <assert.h>
#include <vector>
#include <string>
#include <malloc.h>
#include "caffe/caffe.hpp"
#include "caffe/dnn_batch_test_handler.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"


using namespace caffe;

DNNHandler::DNNHandler(){
  LOG(INFO) << "Initialize the DNNHandler Instance!\n";
  m_device_id = 1;
  m_backend_mode = 1;
}

DNNHandler::DNNHandler( int device_id, int backend_mode ){
  LOG(INFO) << "Initialize the DNNHandler Instance!\n";
  m_device_id = device_id;
  m_backend_mode = backend_mode;
}

DNNHandler::~DNNHandler(){
  reset_engine_status();
  {
    try {
      m_dnn_model.reset();
    } catch(std::exception e) {
      LOG(FATAL) << "Fatal Error: Failed to destory DNNHandler Instance!\n";
    }
  }
  LOG(INFO) << "Finished destory DNNHandler Instance!\n";
}

bool DNNHandler::init_model( std::string net_params, std::string model_path, int backend_mode, int device_id, bool read_from_binary ){

  fprintf( stdout, "NetParams: %s\nTrained Model: %s.\n", net_params.c_str(), model_path.c_str() );

  m_device_id = device_id;
  m_backend_mode = backend_mode;


  if( backend_mode ){
    LOG(INFO) << "Initialize Model to GPU: " << device_id;
    Caffe::set_mode(Caffe::GPU);
    set_GPU_device( device_id );
    //cudaSetDevice( device_id );
  }
  else{
    LOG(INFO) << "Initialize Model to CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  m_dnn_model.reset( new Net<float>( net_params, caffe::TEST ) );

  m_dnn_model->CopyTrainedLayersFrom( model_path );

  get_net_info();
  fprintf(stdout, "Create net done.. [ %s, %d ] \n", m_net_info.net_name.c_str(), m_net_info.data_dim);
 
  return true;
}

void DNNHandler::reset_engine_status(){
  Caffe::SetDevice( m_device_id );  
}

void DNNHandler::get_net_info(){
  m_net_info.net_name = m_dnn_model->name();
  m_net_info.layer_names = m_dnn_model->layer_names();
  m_net_info.blob_names = m_dnn_model->blob_names();
  // get layer type names
  for( int i = 0; i < m_net_info.layer_names.size(); i++ ){
    const boost::shared_ptr< Layer<float> > layer_ = m_dnn_model->layer_by_name(m_net_info.layer_names[i]);
    string layer_type = string(layer_->type());
    m_net_info.layer_type_names.push_back(layer_type);
  }
  // get data info
  vector<Blob<float>*> blob_input_vec = m_dnn_model->input_blobs();
  m_net_info.blob_num = blob_input_vec[0]->num();
  m_net_info.input_blob_channel = blob_input_vec[0]->channels();
  m_net_info.input_blob_width = blob_input_vec[0]->width();
  m_net_info.input_blob_height = blob_input_vec[0]->height();
  m_net_info.data_dim = m_net_info.input_blob_channel*m_net_info.input_blob_width*m_net_info.input_blob_height;

  vector<Blob<float>*> blob_output_vec = m_dnn_model->output_blobs();
  m_net_info.label_dim = blob_output_vec[0]->channels()*blob_output_vec[0]->height()*blob_output_vec[0]->width();
  
}


bool DNNHandler::get_batch_feature( vector< vector<float> >& data_container, vector<string>& layer_names, 
                vector<vector<float> >& batch_features){
  CHECK( layer_names.size() != 0 ) << "the layer_name.size() must not equal to 0";
  int fea_dim = 0;
  for(int i = 0; i < layer_names.size(); i++){
    if ( !m_dnn_model->has_blob(layer_names[i]) ){
      LOG(INFO) << "Layer: " << layer_names[i] << " does not exist in the model, please recheck it!\n";
      return false;
    }
    fea_dim += m_dnn_model->blob_by_name(layer_names[i])->count()/m_dnn_model->blob_by_name(layer_names[i])->num();
  }
  Blob<float>* input_layer = m_dnn_model->input_blobs()[0];
  input_layer->Reshape( data_container.size(), m_net_info.input_blob_channel, m_net_info.input_blob_height, m_net_info.input_blob_width );
  m_dnn_model->Reshape();
 
  //vector2blob
  float* tmp_vec = input_layer->mutable_cpu_data();
  for( int i = 0; i < data_container.size(); i++){
    memcpy(tmp_vec, &data_container[i][0], sizeof(float)*data_container[i].size());
    tmp_vec += data_container[i].size();
  }

  reset_engine_status();
  //LOG( INFO ) << "Forwarding ... ";
  m_dnn_model->Forward();
  for(int i = 0; i < data_container.size(); i++){
    vector<float> fea(fea_dim);
    float* tmpPtr = &fea[0];
    for(int j = 0; j < layer_names.size(); j++){
      int sub_fea_dim = m_dnn_model->blob_by_name(layer_names[j])->count()/m_dnn_model->blob_by_name(layer_names[j])->num();
      memcpy(tmpPtr, m_dnn_model->blob_by_name(layer_names[j])->cpu_data() + i*sub_fea_dim, sizeof(float)*sub_fea_dim);
      tmpPtr += sub_fea_dim;
    }
    batch_features.push_back(fea);
  }
  return true;
}

int DNNHandler::get_blob_width(){
  return m_net_info.input_blob_width;
}

int DNNHandler::get_blob_height(){
  return m_net_info.input_blob_height;
}

int DNNHandler::get_blob_channel(){
  return m_net_info.input_blob_channel;
}

int DNNHandler::get_blob_num(){
  return m_net_info.blob_num;
}

int DNNHandler::get_engine_mode(){
  return m_backend_mode;
}

int DNNHandler::get_device_id(){
  return m_device_id;
}

int DNNHandler::get_GPU_device_count()
{
#ifndef CPU_ONLY  
    int GPU_device_num = 0;
    CUDA_CHECK(cudaGetDeviceCount(&GPU_device_num));
    return GPU_device_num;
#else
    return -1;
#endif
}

int DNNHandler::get_GPU_device()
{
#ifndef CPU_ONLY
    int current_device = 0;
    CUDA_CHECK( cudaGetDevice( &current_device ) );
    return current_device;
#else
    return -1;
#endif
}

void DNNHandler::set_GPU_device( int device_id )
{
#ifndef CPU_ONLY
    CUDA_CHECK(cudaSetDevice( device_id ));
#endif
}

