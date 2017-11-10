#include "caffe/util/frcnn_util.hpp"

namespace caffe {
void BBoxTransformInv(int box_count, const float* box_deltas,
                      const float* pred_cls, const float* boxes, float* pred,
                      int image_height, int image_width, int class_num) {
  float width, height, center_x, center_y;
  float dx, dy, dw, dh;
  float pred_center_x, pred_center_y, pred_width, pred_height;
  for (int n = 0; n < box_count; n++) {
    width = boxes[n * 4 + 2] - boxes[n * 4 + 0] + 1.0;
    height = boxes[n * 4 + 3] - boxes[n * 4 + 1] + 1.0;
    center_x = boxes[n * 4 + 0] + width * 0.5;
    center_y = boxes[n * 4 + 1] + height * 0.5;

    for (int cls = 1; cls < class_num; cls++) {
      dx = box_deltas[(n * class_num + cls) * 4 + 0];
      dy = box_deltas[(n * class_num + cls) * 4 + 1];
      dw = box_deltas[(n * class_num + cls) * 4 + 2];
      dh = box_deltas[(n * class_num + cls) * 4 + 3];
      pred_center_x = center_x + width * dx;
      pred_center_y = center_y + height * dy;
      pred_width = width * std::exp(dw);
      pred_height = height * std::exp(dh);

      pred[(cls * box_count + n) * 5 + 0] = std::max(
          std::min(float(pred_center_x - 0.5 * pred_width), float(image_width - 1)),
          0.0f);
      pred[(cls * box_count + n) * 5 + 1] = std::max(
          std::min(float(pred_center_y - 0.5 * pred_height), float(image_height - 1)),
          0.0f);
      pred[(cls * box_count + n) * 5 + 2] = std::max(
          std::min(float(pred_center_x + 0.5 * pred_width), float(image_width - 1)),
          0.0f);
      pred[(cls * box_count + n) * 5 + 3] = std::max(
          std::min(float(pred_center_y + 0.5 * pred_height), float(image_height - 1)),
          0.0f);
      pred[(cls * box_count + n) * 5 + 4] = pred_cls[n * class_num + cls];
    }
  }
}

void ApplyNMS(vector<vector<float>>& pred_boxes, vector<float>& confidence,
              float nms_thresh) {
  for (int i = 0; i < pred_boxes.size() - 1; i++) {
    float s1 = (pred_boxes[i][2] - pred_boxes[i][0] + 1.0) *
               (pred_boxes[i][3] - pred_boxes[i][1] + 1.0);
    for (int j = i + 1; j < pred_boxes.size(); j++) {
      float s2 = (pred_boxes[j][2] - pred_boxes[j][0] + 1.0) *
                 (pred_boxes[j][3] - pred_boxes[j][1] + 1.0);

      float x1 = std::max(pred_boxes[i][0], pred_boxes[j][0]);
      float y1 = std::max(pred_boxes[i][1], pred_boxes[j][1]);
      float x2 = std::max(pred_boxes[i][2], pred_boxes[j][2]);
      float y2 = std::max(pred_boxes[i][3], pred_boxes[j][3]);
      float width = x2 - x1;
      float height = y2 - y1;
      if (width > 0 && height > 0) {
        float IOU = (width * height) / (s1 + s2 - width * height);
        if (IOU > nms_thresh) {
          if (confidence[i] >= confidence[j]) {
            pred_boxes.erase(pred_boxes.begin() + j);
            confidence.erase(confidence.begin() + j);
            j--;
          } else {
            pred_boxes.erase(pred_boxes.begin() + i);
            confidence.erase(confidence.begin() + i);
            i--;
            break;
          }
        }
      }
    }
  }
}
}
