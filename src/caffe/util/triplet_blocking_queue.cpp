#include <boost/thread.hpp>
#include <string>

#include "caffe/triplet_data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/triplet_blocking_queue.hpp"

namespace caffe {

template<typename T>
class TripletBlockingQueue<T>::sync {
 public:
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
};

template<typename T>
TripletBlockingQueue<T>::TripletBlockingQueue()
    : sync_(new sync()) {
}

template<typename T>
void TripletBlockingQueue<T>::push(const T& t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  queue_.push(t);
  lock.unlock();
  sync_->condition_.notify_one();
}

template<typename T>
bool TripletBlockingQueue<T>::try_pop(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T TripletBlockingQueue<T>::pop(const string& log_on_wait) {
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
bool TripletBlockingQueue<T>::try_peek(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  return true;
}

template<typename T>
T TripletBlockingQueue<T>::peek() {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    sync_->condition_.wait(lock);
  }

  return queue_.front();
}

template<typename T>
size_t TripletBlockingQueue<T>::size() const {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  return queue_.size();
}

template class TripletBlockingQueue<Batch<float>*>;
template class TripletBlockingQueue<Batch<double>*>;
template class TripletBlockingQueue<TripletDatum*>;
template class TripletBlockingQueue<shared_ptr<TripletDataReader::QueuePair> >;
template class TripletBlockingQueue<P2PSync<float>*>;
template class TripletBlockingQueue<P2PSync<double>*>;

}  // namespace caffe
