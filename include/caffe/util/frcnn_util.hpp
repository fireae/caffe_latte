#include <algorithm>  // std::max
#include <cmath>
#include <vector>
namespace caffe {
using std::vector;
void BBoxTransformInv(int box_count, const float* box_deltas,
                      const float* pred_cls, const float* boxes, float* pred,
                      int image_height, int image_width, int class_num);
void ApplyNMS(vector<vector<float> >& pred_boxes, vector<float>& confidence,
              float nms_thresh);
}