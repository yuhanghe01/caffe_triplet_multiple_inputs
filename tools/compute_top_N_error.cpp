/*Author: Yuhang He
 *Date: August 18, 2016
 *Email: yuhanghe@whu.edu.cn
 *Note: this script is implemented for computing the top-5 error rate for image feature retrieval
 */

//ASC indicate sorting in ascending order;
//DESC indicates sorting in descending order;
#include<iostream>
#include<algorithm>
#include<vector>
#include<string>
#include<utility>
#include<fstream>
#include<sstream>
#include"caffe/caffe.hpp"
#include<opencv2/opencv.hpp>


enum CompMode{ ASC, DESC };
struct FeatFormat{
  std::string img_name;
  std::pair< int, std::vector< float > > label_feat;
};
struct TriplePair{
  std::string img_name;
  int label;
  float dist;
};

class Compare{
  private:
    CompMode comp_mode_;
  public:
    Compare( CompMode comp_mode ): comp_mode_( comp_mode ){};
    bool operator () ( std::pair<int, float>& pair1, std::pair<int, float>& pair2 ){
      switch( comp_mode_ ){
        case ASC:
          return pair1.second < pair2.second;
        case DESC:
          return pair1.second > pair2.second;
      }
    } 
};

float comp_dot_product_dist( std::vector<float>& a, std::vector<float>& b){

  CHECK( a.size() > 0 ) << "the input float pointer a should not be null";
  CHECK( b.size() > 0 ) << "the input float pointer b should not be null";

  CHECK( a.size() == b.size() ) << "the two array should be the same size";
  
  int num = a.size();

  float dist = .0;
  for( int i = 0; i < num; ++i ){
    dist += a[i]*b[i];
  }

  return dist;
}



void split_str( const std::string& str_to_split, char delim, std::vector< std::string >& str_split_result ) {
  str_split_result.clear();
  std::stringstream ss( str_to_split );
  std::string item;
  while( std::getline( ss, item, delim ) ) {
      str_split_result.push_back( item );
  }
}

bool read_file( std::string& file_name, std::vector< struct FeatFormat >& file_readed ){
  std::ifstream file_id( file_name.c_str() );
  std::string line_tmp = "";
  while( std::getline( file_id, line_tmp ) ){
    std::vector< std::string > split_rst;
    split_str( line_tmp, ' ', split_rst );
    std::string img_name = split_rst[0];
    int label = 0;
    std::stringstream ss;
    ss << split_rst[1];
    ss >> label;
    std::vector<float> fea_vec_tmp;
    for( int i = 2; i < split_rst.size(); ++i ){
      std::stringstream ss_tmp;
      ss_tmp << split_rst[i];
      float val_tmp = .0;
      ss_tmp >> val_tmp;
      fea_vec_tmp.push_back( val_tmp );   
    }
    struct FeatFormat feat_format_tmp;
    feat_format_tmp.img_name = img_name;
    feat_format_tmp.label_feat = std::make_pair( label, fea_vec_tmp );
    file_readed.push_back( feat_format_tmp );
  }
  
  return true; 
}

std::vector< struct TriplePair >  comp_single_dist( struct FeatFormat& query_fea, std::vector< struct FeatFormat >& base_fea ){
  std::vector< struct TriplePair > label_dist_pair;
  for( int i = 0; i < base_fea.size(); ++i ){
    int label_tmp = base_fea[i].label_feat.first;
    float dist_tmp = .0;
    dist_tmp = comp_dot_product_dist( query_fea.label_feat.second, base_fea[i].label_feat.second );
    struct TriplePair triple_pair_tmp;
    triple_pair_tmp.img_name = base_fea[i].img_name;
    triple_pair_tmp.label = label_tmp;
    triple_pair_tmp.dist = dist_tmp;
    label_dist_pair.push_back( triple_pair_tmp );
  }

  return label_dist_pair;
}

//struct TriplePair{
//  std::string img_name;
//  int label;
//  float dist;
//};

//typedef std::pair<int, float> num_dis;
struct CmpByDis{
  bool operator() (const struct TriplePair& lhs, const struct TriplePair& rhs ){
    return lhs.dist > rhs.dist;
  }
};

std::vector< struct TriplePair > retrieval_top5( int label, std::vector< struct TriplePair >& dist_rst  ){
  std::sort( dist_rst.begin(), dist_rst.end(), CmpByDis() );
  std::vector< struct TriplePair > min_labels;
  for( int i = 0; i < 5; ++i ){
    min_labels.push_back( dist_rst[i] );
  }
  
  return min_labels;
}

struct RetriResult{
  std::string img_name;
  int label;
  std::vector< struct TriplePair > retri_rst;
};

std::vector< struct RetriResult > retrieval_top5_total( std::vector< struct FeatFormat  >& query_fea_list, std::vector< struct FeatFormat >& base_fea_list ){
  std::vector< struct RetriResult > retrieval_rst;
  for( int i = 0; i < query_fea_list.size(); ++i ){
    std::vector< struct TriplePair > dist_vec;
    dist_vec = comp_single_dist( query_fea_list[i], base_fea_list );
    std::vector< struct TriplePair > top5_tmp;
    top5_tmp = retrieval_top5( query_fea_list[i].label_feat.first, dist_vec );
    struct RetriResult retri_result_tmp;
    retri_result_tmp.img_name = query_fea_list[i].img_name;
    retri_result_tmp.label = query_fea_list[i].label_feat.first;
    retri_result_tmp.retri_rst = top5_tmp;
    retrieval_rst.push_back( retri_result_tmp );
  }

  return retrieval_rst;
}

struct top_stat{
  int label;
  int label_num;
  int top_num[5];
  //int top2_num;
  //int top3_num;
  //int top4_num;
  //int top5_num;
};

std::vector< std::pair< int, std::vector<float> > > comp_top5_error( std::vector< struct RetriResult > retrieval_rst ){
  std::vector< struct top_stat > top_stat_vec(100);
  for( int i = 0; i < top_stat_vec.size(); ++i ){
    top_stat_vec[i].label = 0;
    top_stat_vec[i].label_num = 0;
    top_stat_vec[i].top_num[0] = 0;
    top_stat_vec[i].top_num[1] = 0;
    top_stat_vec[i].top_num[2] = 0;
    top_stat_vec[i].top_num[3] = 0;
    top_stat_vec[i].top_num[4] = 0;
  }
  for( int i = 0; i < retrieval_rst.size(); ++i ){
    top_stat_vec[ retrieval_rst[i].label ].label = retrieval_rst[i].label;
    top_stat_vec[ retrieval_rst[i].label ].label_num = 
    top_stat_vec[ retrieval_rst[i].label ].label_num + 1;
    int label = retrieval_rst[i].label;
    //std::vector<int> top_rst = retrieval_rst[i].second;
    std::vector<int> top_rst;
    for( int j = 0; j < 5; ++j ){
      top_rst.push_back( retrieval_rst[i].retri_rst[j].label );
    }
    for( int j = 0; j < 5; ++j ){
      //we only consider the the top-k is right if top-(0,k-1) holds the label;
      if( j == 0 && top_rst[j] == label ){
        top_stat_vec[ label ].top_num[j] = 
        top_stat_vec[ label ].top_num[j] + 1;
      }
      else{
        for( int k = 0; k < j; k++ ){
          if( top_rst[k] == label ){
            top_stat_vec[ label ].top_num[j] = 
            top_stat_vec[ label ].top_num[j] + 1;
            break;
          }
        } 
      }  
    }   
  }
  std::vector< std::pair< int, std::vector<float> > > top_rst;
  for( int i = 0; i < 100; ++i ){
    if( top_stat_vec[i].label_num == 0 )
      continue;
    int label = top_stat_vec[i].label;
    std::vector<float> top_rst_tmp;
    for( int j = 0; j < 5; ++j ){
      CHECK( top_stat_vec[i].label_num > 0 ) << "the label_num should be larger than 0";
      int num_tmp = top_stat_vec[i].top_num[j];
      //for( int k = 0; k < j; k++ ){
      //  num_tmp += top_stat_vec[i].top_num[k];
      //}
      
      //top_rst_tmp.push_back( float(top_stat_vec[i].top_num[j])/float( top_stat_vec[i].label_num ) );
      top_rst_tmp.push_back( float(num_tmp)/float( top_stat_vec[i].label_num ) );
    }
    top_rst.push_back( std::make_pair( label, top_rst_tmp ) );
  }

  return top_rst; 
}

/*
std::pair< int, std::vector<float> > comp_total( std::vector< std::pair<int, std::vector<float> > >& query_list, std::vector< std::pair<int, std::vector<float> > >& base_list ){
  
  for(int i = 0; i < query_list.size(); ++i ){
    std::vector< std::pair<int, float> > dist_tmp;
    dist_tmp = comp_single_dist( query_list[i], base_list );
    std::pair<int, std::vector<int> > top5_rst;
    top5_rst = retrieval_top5( query_list[i].first, dist_tmp  );
    
  }


}
*/

DEFINE_string( query_fea_list, "",
   "the query image feature list.");
DEFINE_string( base_fea_list, "",
   "the base image feature list.");
DEFINE_string( save_dir, "",
   "the directory to save the sample retrieved images"); 
//DEFINE_int( save_num, 0,
//   "the number of retrieved sample images to be saved");
//struct RetriLabelFormat{
//  std::string img_name;
//  std::pair< int, std::vector<int> > label_retrilabel;
//};
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

  std::string query_fea_list = FLAGS_query_fea_list;
  std::string base_fea_list = FLAGS_base_fea_list;
  std::string save_dir = FLAGS_save_dir;

  CHECK( query_fea_list.size() > 0 ) << "the input query_fea_list size must be larger then 0";
  CHECK( base_fea_list.size() > 0 ) << "the input base_fea_list size must be larger then 0";


  std::vector< struct FeatFormat > query_list;   
  std::vector< struct FeatFormat > base_list;

  read_file( query_fea_list, query_list );   
  read_file( base_fea_list, base_list );

  //LOG( INFO ) << "query_list size is " << query_list.size(); 
  //LOG( INFO ) << "base_list size is " << base_list.size();

  std::vector< struct RetriResult > top5_rst;
  top5_rst = retrieval_top5_total( query_list, base_list );
  //std::string save_dir = "/mnt/lvm/heyuhang/taobao_pair_train/tools/test_img_new/sample_img/";
  for( int i = 0; i < 100; ++i ){
    struct RetriResult retri_result = top5_rst[i];
    //if( retri_result.label != 6 && retri_result.label != 15 )
    //  continue;
    std::stringstream ss;
    ss << i;
    std::string img_name;
    ss >> img_name;
    std::string save_name_query = save_dir + img_name + ".jpg";
    cv::Mat query_img = cv::imread( retri_result.img_name, CV_LOAD_IMAGE_COLOR );
    if( !query_img.data )
      continue;
    cv::imwrite( save_name_query, query_img );
    for( int j = 0; j < 5; ++j ){
      std::string img_name_tmp = retri_result.retri_rst[j].img_name;
      cv::Mat retri_img = cv::imread( img_name_tmp, CV_LOAD_IMAGE_COLOR );
      if( !retri_img.data )
        continue;
      std::stringstream ss_tmp;
      std::string save_img_name_tmp;
      ss_tmp << j;
      ss_tmp >> save_img_name_tmp;
      std::string save_name_retri = save_dir + img_name + "_" + save_img_name_tmp + ".jpg";
      cv::imwrite( save_name_retri, retri_img );
    } 
  }
  LOG(INFO) << "top5_rst size is " << top5_rst.size();
  //LOG( INFO ) << "top5_rst[0] " << top5_rst[0].first << "\n" 
  //            << "top5_rst[0][0] " << top5_rst[0].second[0] << "\n"
  //            << "top5_rst[0][1] " << top5_rst[0].second[1] << "\n"
  //            << "top5_rst[0][2] " << top5_rst[0].second[2] << "\n"
  //            << "top5_rst[0][3] " << top5_rst[0].second[3] << "\n"
  //            << "top5_rst[0][4] " << top5_rst[0].second[4] << "\n";
  std::vector< std::pair<int, std::vector<float> > > top5_final_rst;
  top5_final_rst = comp_top5_error( top5_rst );

  LOG(INFO) << "top5_final_rst size() is " << top5_final_rst.size();
  for( int i = 0; i < top5_final_rst.size(); ++i ){
    std::cout << "label = " << top5_final_rst[i].first << "\n"
              << "top1 = " << top5_final_rst[i].second[0] << "\n"  
              << "top2 = " << top5_final_rst[i].second[1] << "\n"  
              << "top3 = " << top5_final_rst[i].second[2] << "\n"  
              << "top4 = " << top5_final_rst[i].second[3] << "\n"  
              << "top5 = " << top5_final_rst[i].second[4] << std::endl;  
  }     

  return 0;
}
