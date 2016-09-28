#include <boost/thread.hpp>
#include <string>

#include "caffe/triplet_multiple_data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/triplet_multiple_blocking_queue.hpp"

namespace caffe {

template<typename T>
class TripletMultipleBlockingQueue<T>::sync {
 public:
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
};

template<typename T>
TripletMultipleBlockingQueue<T>::TripletMultipleBlockingQueue()
    : sync_(new sync()) {
}

template<typename T>
void TripletMultipleBlockingQueue<T>::push(const T& t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  queue_.push(t);
  lock.unlock();
  sync_->condition_.notify_one();
}

template<typename T>
bool TripletMultipleBlockingQueue<T>::try_pop(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T TripletMultipleBlockingQueue<T>::pop(const string& log_on_wait) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    if (!log_on_wait.empty()) {
      LOG_EVERY_N(INFO, 1000)<< log_on_wait;
    }
    sync_->condition_.wait(lock);
  }

  T t = queue_.front();
  queue_.pop();
  return t;
}

template<typename T>
bool TripletMultipleBlockingQueue<T>::try_peek(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  return true;
}

template<typename T>
T TripletMultipleBlockingQueue<T>::peek() {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    sync_->condition_.wait(lock);
  }

  return queue_.front();
}

template<typename T>
size_t TripletMultipleBlockingQueue<T>::size() const {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  return queue_.size();
}

template class TripletMultipleBlockingQueue<Batch<float>*>;
template class TripletMultipleBlockingQueue<Batch<double>*>;
template class TripletMultipleBlockingQueue<TripletMultipleDatum*>;
template class TripletMultipleBlockingQueue<shared_ptr<TripletMultipleDataReader::QueuePair> >;
template class TripletMultipleBlockingQueue<P2PSync<float>*>;
template class TripletMultipleBlockingQueue<P2PSync<double>*>;

}  // namespace caffe
