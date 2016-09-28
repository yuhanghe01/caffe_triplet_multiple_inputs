// This program converts a triplet list to DB format as Triplet_Datum proto buffers
/*
 *
 * Author: Yuhang He 
 * Email: heyuhang@dress-plus.com
 * Date: Aug. 11, 2016
 */

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");
DEFINE_string(triplet_list_name,"",
    "Required: the triplet list file, in which each line stores the anchor/positive/negative images, respectively, being separated by \t or a blank.");
DEFINE_string(db_save_name,"",
    "Required: the file name that stores the created DB proto buffers.");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images in triplet format to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_triplet_dataset [FLAGS]\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 1) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_triplet_datum");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  const std::string triplet_list_name = FLAGS_triplet_list_name;
  const std::string db_save_name = FLAGS_db_save_name;

  CHECK( triplet_list_name.size() > 0 ) << "the triplet_list_name param should be specified!";
  CHECK( db_save_name.size() > 0 ) << "the db_save_name param should be specified!";

  std::ifstream infile( triplet_list_name.c_str() );
  std::vector< std::vector< std::string > > lines;
  std::string line;
  //size_t pos;
  std::string anchor_img_name;
  std::string pos_img_name;
  std::string neg_img_name;
  //std::vector< std::string > triple_pair(3);
  while( infile >> anchor_img_name >> pos_img_name >> neg_img_name ){
    std::vector< std::string > triple_pair;
    triple_pair.push_back( anchor_img_name );
    triple_pair.push_back( pos_img_name );
    triple_pair.push_back( neg_img_name );
    lines.push_back( triple_pair );
  }
  infile.close();
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open( db_save_name.c_str(), db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  //std::string root_folder(argv[1]);
  TripletDatum triplet_datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id][0];
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    status = ReadTripletImagesToTripletDatum( lines[line_id], resize_height, resize_width, is_color, enc, &triplet_datum);
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = triplet_datum.channels() * triplet_datum.height() * triplet_datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = triplet_datum.data_anchor();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id][0];

    // Put in db
    std::string out;
    CHECK( triplet_datum.SerializeToString( &out ) );
    txn->Put( key_str, out );

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
