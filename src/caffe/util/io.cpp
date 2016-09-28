#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    //LOG(ERROR) << "Could not open or find file " << filename;
    LOG( INFO ) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    int img_height = cv_img_origin.rows;
    int img_width = cv_img_origin.cols;
    if( img_height != img_width ){
      int img_new_length = img_width > img_height?img_width:img_height; 
      int offset = int(( img_new_length + img_new_length - img_width - img_height )/2.0);
      cv::Mat img_new;
      if( img_width > img_height ){
       cv::copyMakeBorder( cv_img_origin, img_new, offset, offset, 0, 0, cv::BORDER_REPLICATE );
      }
      else{
        cv::copyMakeBorder( cv_img_origin, img_new, 0, 0, offset, offset, cv::BORDER_REPLICATE );
      }
    cv::resize( img_new, cv_img, cv::Size(width, height) );
  }
  else
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }

  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadMultipleTripletImagesToMultipleTripletDatum( const struct MultipleTripletPair& mult_triplet_img_list, const int height, const int width, const bool is_color, const std::string& encoding, TripletMultipleDatum* mult_triplet_datum){

  //CHECK( triplet_img_list.size() == 3 ) << "the triple_img_list size should be 3 (anchor/pos/neg respectively)";
  std::vector< cv::Mat > anchor_img_vec;
  std::vector< cv::Mat > pos_img_vec;
  std::vector< cv::Mat > neg_img_vec;
  //Anchor image vector
  for( int i = 0; i < mult_triplet_img_list.anchor_vec.size(); ++i ){
    cv::Mat cv_img_tmp = cv::imread( mult_triplet_img_list.anchor_vec[i], is_color );
    if( !cv_img_tmp.data ){
        return false;
    }
    anchor_img_vec.push_back( cv_img_tmp );
  }
  //Positive image vector
  for( int i = 0; i < mult_triplet_img_list.pos_vec.size(); ++i ){
    cv::Mat cv_img_tmp = cv::imread( mult_triplet_img_list.pos_vec[i], is_color );
    if( !cv_img_tmp.data ){
        return false;
    }
    pos_img_vec.push_back( cv_img_tmp );
  }
  //Negative image vector
  for( int i = 0; i < mult_triplet_img_list.neg_vec.size(); ++i ){
    cv::Mat cv_img_tmp = cv::imread( mult_triplet_img_list.neg_vec[i], is_color );
    if( !cv_img_tmp.data ){
        return false;
    }
    neg_img_vec.push_back( cv_img_tmp );
  }


  /*
  cv::Mat anchor_img = ReadImageToCVMat( triplet_img_list[0], height, width, is_color );
  cv::Mat pos_img = ReadImageToCVMat( triplet_img_list[1], height, width, is_color );
  cv::Mat neg_img = ReadImageToCVMat( triplet_img_list[2], height, width, is_color );

  if( !anchor_img.data || !pos_img.data || !neg_img.data ){
    LOG(INFO) << "the three images cannot to successfully loaded simultaneously!";
    return false;
  }
  */
  if( encoding.size() ){
    //serialize the anchor image list;
    //std::vector<uchar> buf_anchor;
    //std::string anchor_str;
    for( int i = 0; i < anchor_img_vec.size(); ++i ){
      std::vector<uchar> buf_anchor;
      std::string anchor_str;
      cv::imencode("." + encoding, anchor_img_vec[i], buf_anchor );
      anchor_str = std::string( reinterpret_cast<char*>(&buf_anchor[0]), buf_anchor.size() );
      mult_triplet_datum -> add_data_anchor( anchor_str ); 
    }

    for( int i = 0; i < pos_img_vec.size(); ++i ){
      std::vector<uchar> buf_pos;
      std::string pos_str;
      cv::imencode("." + encoding, pos_img_vec[i], buf_pos );
      pos_str = std::string( reinterpret_cast<char*>(&buf_pos[0]), buf_pos.size() );
      mult_triplet_datum -> add_data_pos( pos_str ); 
    }

    for( int i = 0; i < neg_img_vec.size(); ++i ){
      std::vector<uchar> buf_neg;
      std::string neg_str;
      cv::imencode("." + encoding, neg_img_vec[i], buf_neg );
      neg_str = std::string( reinterpret_cast<char*>(&buf_neg[0]), buf_neg.size() );
      mult_triplet_datum -> add_data_neg( neg_str ); 
    }
    //mult_triplet_datum -> set_channels
    mult_triplet_datum -> set_encoded( true );
    //std::vector<uchar> buf_pos;
    //std::vector<uchar> buf_neg;
    
   // cv::imencode("." + encoding, anchor_img, buf_anchor );
   // cv::imencode("." + encoding, pos_img, buf_pos );
   // cv::imencode("." + encoding, neg_img, buf_neg );

    //mult_triplet_datum -> set_data_anchor( std::string( reinterpret_cast<char*>(&buf_anchor[0]), buf_anchor.size() ) );
    //mult_triplet_datum -> set_data_pos( std::string( reinterpret_cast<char*>(&buf_pos[0]), buf_pos.size() ) );
    //mult_triplet_datum -> set_data_neg( std::string( reinterpret_cast<char*>(&buf_neg[0]), buf_neg.size() ) );
    //mult_triplet_datum -> set_encoded( true );
    //note that if these images are encoded, we don't need to set the channels, height, width param.  
    return true; 
  }
  else
    return false;
  //if not encoded we need to create triplet_datum by hand
  /*
  CHECK( anchor_img.depth() == CV_8U ) << " anchor image data type must be unsigned byte";
  CHECK( pos_img.depth() == CV_8U ) << " positive image data type must be unsigned byte";
  CHECK( neg_img.depth() == CV_8U ) << " negative image data type must be unsigned byte";
  CHECK( anchor_img.channels() == pos_img.channels() && anchor_img.channels() == neg_img.channels() )
       << "the anchor/pos/neg images should have the same channels number";
  CHECK( anchor_img.rows == pos_img.rows && anchor_img.cols == pos_img.cols &&
         anchor_img.rows == neg_img.rows && anchor_img.cols == neg_img.cols ) <<
       "The anchor/positive/negative images should share the same rows and cols";
  triplet_datum->set_channels( anchor_img.channels() );
  triplet_datum->set_height(anchor_img.rows);
  triplet_datum->set_width(anchor_img.cols);
  triplet_datum->clear_data_anchor();
  triplet_datum->clear_data_pos();
  triplet_datum->clear_data_neg();
  //triplet_datum->clear_float_data();
  triplet_datum->set_encoded( false );
  int triplet_datum_channels = triplet_datum->channels();
  int triplet_datum_height = triplet_datum->height();
  int triplet_datum_width = triplet_datum->width();
  int triplet_datum_size = triplet_datum_channels * triplet_datum_height * triplet_datum_width;
  std::string buffer_anchor( triplet_datum_size, ' ' );
  std::string buffer_pos( triplet_datum_size, ' ' );
  std::string buffer_neg( triplet_datum_size, ' ' );
  for (int h = 0; h < triplet_datum_height; ++h) {
    const uchar* ptr_anchor = anchor_img.ptr<uchar>(h);
    const uchar* ptr_pos = pos_img.ptr<uchar>(h);
    const uchar* ptr_neg = neg_img.ptr<uchar>(h);
    int anchor_img_index = 0;
    int pos_img_index = 0;
    int neg_img_index = 0;
    for ( int w = 0; w < triplet_datum_width; ++w ) {
      for (int c = 0; c < triplet_datum_channels; ++c) {
        int datum_index = (c * triplet_datum_height + h) * triplet_datum_width + w;
        buffer_anchor[datum_index] = static_cast<char>(ptr_anchor[anchor_img_index++]);
        buffer_pos[datum_index] = static_cast<char>(ptr_pos[pos_img_index++]);
        buffer_neg[datum_index] = static_cast<char>(ptr_neg[neg_img_index++]);
      }
    }
  }
  triplet_datum->set_data_anchor( buffer_anchor );
  triplet_datum->set_data_pos( buffer_pos );
  triplet_datum->set_data_neg( buffer_neg );
  return true;
  */
}



bool ReadTripletImagesToTripletDatum( const std::vector< std::string >& triplet_img_list, const int height, const int width, const bool is_color, const std::string& encoding, TripletDatum* triplet_datum){
  //note that triplet_img_list[0] indicates the anchor img
  //note that triplet_img_list[1] indicates the positve img
  //note that triplet_img_list[2] indicates the negative img
  CHECK( triplet_img_list.size() == 3 ) << "the triple_img_list size should be 3 (anchor/pos/neg respectively)";
  cv::Mat anchor_img = ReadImageToCVMat( triplet_img_list[0], height, width, is_color );
  cv::Mat pos_img = ReadImageToCVMat( triplet_img_list[1], height, width, is_color );
  cv::Mat neg_img = ReadImageToCVMat( triplet_img_list[2], height, width, is_color );

  if( !anchor_img.data || !pos_img.data || !neg_img.data ){
    LOG(INFO) << "the three images cannot to successfully loaded simultaneously!";
    return false;
  }
  if( encoding.size() ){
    std::vector<uchar> buf_anchor;
    std::vector<uchar> buf_pos;
    std::vector<uchar> buf_neg;
    
    cv::imencode("." + encoding, anchor_img, buf_anchor );
    cv::imencode("." + encoding, pos_img, buf_pos );
    cv::imencode("." + encoding, neg_img, buf_neg );

    triplet_datum -> set_data_anchor( std::string( reinterpret_cast<char*>(&buf_anchor[0]), buf_anchor.size() ) );
    triplet_datum -> set_data_pos( std::string( reinterpret_cast<char*>(&buf_pos[0]), buf_pos.size() ) );
    triplet_datum -> set_data_neg( std::string( reinterpret_cast<char*>(&buf_neg[0]), buf_neg.size() ) );
    triplet_datum -> set_encoded( true );
    //note that if these images are encoded, we don't need to set the channels, height, width param.  
    return true; 
  }
  //if not encoded we need to create triplet_datum by hand

  CHECK( anchor_img.depth() == CV_8U ) << " anchor image data type must be unsigned byte";
  CHECK( pos_img.depth() == CV_8U ) << " positive image data type must be unsigned byte";
  CHECK( neg_img.depth() == CV_8U ) << " negative image data type must be unsigned byte";
  CHECK( anchor_img.channels() == pos_img.channels() && anchor_img.channels() == neg_img.channels() )
       << "the anchor/pos/neg images should have the same channels number";
  CHECK( anchor_img.rows == pos_img.rows && anchor_img.cols == pos_img.cols &&
         anchor_img.rows == neg_img.rows && anchor_img.cols == neg_img.cols ) <<
       "The anchor/positive/negative images should share the same rows and cols";
  triplet_datum->set_channels( anchor_img.channels() );
  triplet_datum->set_height(anchor_img.rows);
  triplet_datum->set_width(anchor_img.cols);
  triplet_datum->clear_data_anchor();
  triplet_datum->clear_data_pos();
  triplet_datum->clear_data_neg();
  //triplet_datum->clear_float_data();
  triplet_datum->set_encoded( false );
  int triplet_datum_channels = triplet_datum->channels();
  int triplet_datum_height = triplet_datum->height();
  int triplet_datum_width = triplet_datum->width();
  int triplet_datum_size = triplet_datum_channels * triplet_datum_height * triplet_datum_width;
  std::string buffer_anchor( triplet_datum_size, ' ' );
  std::string buffer_pos( triplet_datum_size, ' ' );
  std::string buffer_neg( triplet_datum_size, ' ' );
  for (int h = 0; h < triplet_datum_height; ++h) {
    const uchar* ptr_anchor = anchor_img.ptr<uchar>(h);
    const uchar* ptr_pos = pos_img.ptr<uchar>(h);
    const uchar* ptr_neg = neg_img.ptr<uchar>(h);
    int anchor_img_index = 0;
    int pos_img_index = 0;
    int neg_img_index = 0;
    for ( int w = 0; w < triplet_datum_width; ++w ) {
      for (int c = 0; c < triplet_datum_channels; ++c) {
        int datum_index = (c * triplet_datum_height + h) * triplet_datum_width + w;
        buffer_anchor[datum_index] = static_cast<char>(ptr_anchor[anchor_img_index++]);
        buffer_pos[datum_index] = static_cast<char>(ptr_pos[pos_img_index++]);
        buffer_neg[datum_index] = static_cast<char>(ptr_neg[neg_img_index++]);
      }
    }
  }
  triplet_datum->set_data_anchor( buffer_anchor );
  triplet_datum->set_data_pos( buffer_pos );
  triplet_datum->set_data_neg( buffer_neg );
  return true;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
#endif  // USE_OPENCV
}  // namespace caffe
