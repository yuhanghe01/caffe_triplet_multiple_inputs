/*@Author: Yuhang He
 *@Email: yuhanghe@whu.edu.cn
 *@Date: Mar. 28, 2016
 *@Note: header file designed for caffe-based image test in batch size
 */
#ifndef DNN_BATCH_TEST_HANDLER_H
#define	DNN_BATCH_TEST_HANDLER_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include <caffe/caffe.hpp>

using namespace caffe;
//using cv::Mat;

struct NetInfo
{
  std::string net_name;
  std::vector<std::string> layer_names;
  std::vector<std::string> blob_names;
  std::vector<std::string> layer_type_names;
  int data_dim;
  int label_dim;
  int blob_num;
  int input_blob_channel;
  int input_blob_width;
  int input_blob_height;
};

class DNNHandler
{
  public:
    DNNHandler();
    DNNHandler( int device_id = 1, int backend_mode = 1 );
    ~DNNHandler();
    NetInfo m_net_info;
    bool init_model( std::string net_params, std::string model_path, int backend_mode, int device_id = 0, bool read_from_binary = false );
    void get_net_info();
    bool get_batch_feature( std::vector< std::vector<float> >& data_container, std::vector<std::string>& layer_names, std::vector<std::vector<float> >& batch_features );
    int get_blob_width();
    int get_blob_height();
    int get_blob_channel();
    int get_blob_num();
    int get_engine_mode();
    int get_device_id();
    static int get_GPU_device_count();
    static int get_GPU_device();
    static void set_GPU_device( int device_id );

  private:
    //void convt_Mat2_vec( cv::Mat input_img, std::vector<float>& v);
    void reset_engine_status();
    boost::shared_ptr< Net<float> > m_dnn_model;
    int m_device_id;
    int m_backend_mode;
};

#endif


