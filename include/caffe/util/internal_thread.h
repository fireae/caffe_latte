#ifndef CAFFE_UTIL_THREAD_HPP_
#define CAFFE_UTIL_THREAD_HPP_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace caffe {

struct thread_interrupted {};

class InterruptionPoint {
 public:
  InterruptionPoint() : stop(false) {}
  void Interrupt() {
    std::unique_lock<std::mutex> lock(mutex);
    stop = true;
  }
  void InterruptionRequested() {
    std::unique_lock<std::mutex> lock(mutex);
    if (stop) throw thread_interrupted();
  }

 protected:
  bool stop;
  std::mutex mutex;
  std::condition_variable cond;
};

class InternalThread {
 public:
  ~InternalThread() { StopInternalThread(); }
  void StartInternalThread() {
    thread = std::unique_ptr<std::thread>(
        new std::thread(std::bind(&InternalThread::InternalThreadEntry, this)));
  }
  void StopInternalThread() {
    interruption_point.Interrupt();
    thread->join();
  }
  bool must_stop() {
    interruption_point.InterruptionRequested();
    return false;
  }

 protected:
  virtual void InternalThreadEntry() {
    try {
      while (!must_stop()) {
      }
    } catch (const thread_interrupted&) {
    }
  }

 private:
  std::unique_ptr<std::thread> thread;
  InterruptionPoint interruption_point;
};

}  // namespace caffe

#endif  // CAFFE_UTIL_THREAD_HPP_