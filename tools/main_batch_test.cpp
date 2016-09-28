/*@Author: Yuhang He
 *@Date: Mar. 29, 2016
 *@Email: yuhanghe@whu.edu.cn
 */

#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>

#include <algorithm>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "caffe/dnn_batch_test_handler.hpp"



void convert_Mat2_vec( cv::Mat& input_img, std::vector<float>& img_vec, float mean_val ){
  img_vec.clear();
  img_vec.resize( input_img.cols * input_img.rows * input_img.channels() );

  //LOG(INFO) << "in convert_Mat2_vec function, mean_val = " << mean_val;
  int step_size = input_img.cols*input_img.rows;
  for ( int i = 0; i < input_img.rows; ++i ){
    for ( int j = 0; j < input_img.cols; ++j ){
      cv::Vec3b intensity = input_img.at< cv::Vec3b >(i, j);
      for (int c = 0; c < input_img.channels(); ++c){
        img_vec[ i * input_img.cols + j + step_size * c ] = intensity.val[c] - mean_val;
      }
    }
  }
}

void split_str( const std::string& str_to_split, char delim, std::vector< std::string >& str_split_result ) {
  str_split_result.clear();
  std::stringstream ss( str_to_split );
  std::string item;
  while( std::getline( ss, item, delim ) ) {
      str_split_result.push_back( item );
  }
}

cv::Mat add_border_and_resize( cv::Mat& input_img, int crop_size ){
  //LOG(INFO) << "in add_border_and_resize function, crop_size = " << crop_size;

  cv::Mat cv_img;
  int w = input_img.size().width;
  int h = input_img.size().height;
  int max_border = w > h?w:h;
  cv::Mat zero_tmp = cv::Mat::zeros(max_border, max_border, CV_8UC3);
  if(w > h)
    cv_img = zero_tmp(cv::Rect(0, w/2-h/2, w, h));
  else
    cv_img = zero_tmp(cv::Rect(h/2-w/2, 0, w, h));
  input_img.copyTo(cv_img);
  cv::resize(zero_tmp, cv_img, cv::Size( crop_size, crop_size ));
 
  //cv::imwrite( "crop_rest.jpg", cv_img );
  return cv_img; 
}

DEFINE_string( net_param, "",
    "caffe net layer param used for testing.");

DEFINE_string( model_path, "",
    "caffe trained model directory and name.");

DEFINE_string( img_path, "./",
        "img_to_test directory.");

DEFINE_string( img_list_file, "",
        "img list file path");

DEFINE_string( layer_name, "prob",
        "the layer name to extract feature");

DEFINE_string( output_file, "result.txt",
        "output_file used to store the result");

DEFINE_int32( backend_mode, 1,
        "backend_mode used to test the image, 0 for CPU and 1 for GPU.");

DEFINE_int32( device_id, 0, 
        "device_id used for computation: >1: GPU number, 0: CPU");

DEFINE_double( mean_val, 128.0,
        "the mean value of the test image, default is 128.0");

int main ( int argc, char **argv ){

  ::google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
 
  #ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
  #endif
  gflags::SetUsageMessage("test a bunch of images in batch size with caffe pre-trained model\n"
        "format used as:\n"
        "Usage:\n"
        "main_batch_test net_param model_path img_path img_list_file layer_name output_file backend_mode device_id mean_val");

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if ( argc != 1 ){
    std::cout << "argc = " << argc << std::endl;
    gflags::ShowUsageWithFlagsRestrict(argv[0], "main_batch_test");  
    return 1;
  }

  //parse the argument
  std::string net_param = FLAGS_net_param;
  std::string model_path = FLAGS_model_path;
  std::string img_path = FLAGS_img_path;
  std::string img_list_file = FLAGS_img_list_file;
  std::string layer_name = FLAGS_layer_name;
  std::string output_file = FLAGS_output_file; 
  int backend_mode = FLAGS_backend_mode; 
  int device_id = FLAGS_device_id;
  float mean_val = float(FLAGS_mean_val);
 
  DNNHandler dnn_handler( device_id, backend_mode );
  dnn_handler.init_model( net_param, model_path, backend_mode, device_id, false ); 

  std::ifstream image_list_file;
  image_list_file.open( img_list_file.c_str() );
  if( !image_list_file.is_open() ){
    LOG(ERROR) << "Failed to open image list file : " << img_list_file;
  }

  std::vector< std::string > layer_names;
  split_str( layer_name, ',', layer_names );
  CHECK( layer_names.size() > 0 ) << "the input img_list_file should have at least one line!\n";

  std::ofstream output_file_to_write( output_file.c_str() );
  bool readline = true;
  std::string line;
  int batch_index = 1;

  int batch_size = dnn_handler.get_blob_num();
  LOG( INFO ) << "batch_size = " << batch_size;
  while( readline ){
      std::vector < std::vector < float > > data_container;
      std::vector < std::string > file_names;
      std::vector < std::string > label_names;
      LOG(INFO) << "Processing the batch: " << batch_index;

      batch_index += 1;
      int i = 0;
      for(;i < batch_size;){
          readline = std::getline( image_list_file, line );
          if( !readline ) 
              break;
          std::vector< std::string > fea_vec;
          split_str( line, ' ', fea_vec );
          //LOG( INFO ) << "fea_vec size = " << fea_vec.size();
          //if( fea_vec.size() == 2 )
          //  LOG( INFO ) << "fea_vec[1] = " << fea_vec[1];

          std::string img_dir_tmp = "";
          if( img_path[ img_path.size() - 1 ] == '/' )
            img_dir_tmp = img_path + fea_vec[0];
          else
            img_dir_tmp = img_path + "/" + fea_vec[0];

          //LOG(INFO) << "img_dir_tmp = " << img_dir_tmp;
          img_dir_tmp = fea_vec[0];
          //LOG(INFO) << "img_dir_tmp = " << img_dir_tmp;
          cv::Mat patch_img = cv::imread( img_dir_tmp, CV_LOAD_IMAGE_COLOR );
   
          if( !patch_img.data ){
            LOG( INFO ) << "Cannot read the data!";
            continue;
          }
          if( fea_vec.size() > 2 ){
            cv::Rect roi_rect = cv::Rect( int(atof(fea_vec[1].c_str())), int(atof(fea_vec[2].c_str())), int(atof(fea_vec[3].c_str())), int(atof(fea_vec[4].c_str())));
            patch_img = patch_img( roi_rect );
          }

          std::vector< float > data_vector;
          // add border
          int img_crop_size = dnn_handler.get_blob_width();
          cv::Mat border_img = add_border_and_resize( patch_img, img_crop_size );
          convert_Mat2_vec( border_img, data_vector, mean_val );
          data_container.push_back( data_vector );
          //fea_vec[1] here indicates the label;
          file_names.push_back( fea_vec[0] );
          label_names.push_back( fea_vec[1] );
          i++;
          //LOG( INFO ) << "i= " << i;
      } 
      if( data_container.size() == 0 )
          break;
      for( ; i < batch_size; ++i ){
          data_container.push_back( data_container[0] );
          file_names.push_back( "NULL" );
          label_names.push_back( "NULL" );          
          //file_names.push_back( "NULL" );          
      }
      CHECK( data_container.size() == batch_size ) << "data_container size is not equal to batch_size";

      double tt = cvGetTickCount();
      std::vector < std::vector < float > > feature_vec;
      if ( !dnn_handler.get_batch_feature( data_container, layer_names, feature_vec) )
        LOG(ERROR) << "Failed to extract feature in get_batch_feature function";
      tt = ( cvGetTickCount() - tt ) / (1000 * cvGetTickFrequency());
      fprintf( stdout, "Time used: %f ms *** data size:  %d, *** feature size: %d.\n", tt, 
               int( data_container[0].size() ), int( feature_vec[0].size() ) );
     
      CHECK( feature_vec.size() == batch_size && file_names.size() == batch_size && label_names.size() == batch_size );
      for( int ii = 0; ii < feature_vec.size(); ii++ ){
          if ( file_names[ii] == "NULL" )
              continue;
          if( label_names[ii] == "NULL" )
            continue;
          output_file_to_write << file_names[ii];
          output_file_to_write << " " << label_names[ii];
          for ( int jj = 0; jj < feature_vec[ii].size(); jj++ )
              output_file_to_write << " " << feature_vec[ii][jj];
          output_file_to_write << std::endl;
      }
  }

  image_list_file.close();
  output_file_to_write.close();

  return 0;
}
