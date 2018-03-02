#ifndef GRAPH_HPP_
#define GRAPH_HPP_
#include <algorithm>
#include <vector>
#include "cfg.hpp"
// const int kMAX_HORIZONTAL_GAP = 50;
// const float kMIN_V_OVERLAPS = 0.7;
// const float kMIN_SIZE_SIM = 0.7;
namespace jdcn {
using namespace std;
class Graph {
 public:
  Graph(vector<vector<bool>> graph) : graph_(graph) {}

  bool is_any(int index, int type) {
    if (type == 0) {
      for (int i = 0; i < graph_.size(); i++) {
        if (graph_[i][index] == true) {
          return true;
        }
      }
    } else {
      for (int i = 0; i < graph_.size(); i++) {
        if (graph_[index][i] == true) {
          return true;
        }
      }
    }
    return false;
  }

  vector<vector<int>> SubGraphsConnected() {
    int v = 0;
    vector<vector<int>> sub_graphs;
    for (int index = 0; index < graph_.size(); index++) {
      if (!is_any(index, 0) && is_any(index, 1)) {
        v = index;
        vector<int> sub_graph;
        sub_graph.push_back(v);
        sub_graphs.push_back(sub_graph);
        while (is_any(v, 1)) {
          int v_index = 0;
          for (int i = 0; i < graph_.size(); i++) {
            if (graph_[v][i] == true) {
              v_index = i;
            }
          }

          sub_graphs[sub_graphs.size() - 1].push_back(v_index);
          v = v_index;
        }
      }
    }
    return sub_graphs;
  }
  vector<vector<bool>> graph_;
};

class TextProposalGraphBuilder {
 public:
  TextProposalGraphBuilder() {}

  vector<int> get_successions(int index) {
    vector<float> box = text_proposals_[index];
    vector<int> results;
    int min_value =
        std::min(int(box[0]) + kMAX_HORIZONTAL_GAP + 1, im_size_[1]);
    for (int left = int(box[0]) + 1; left < min_value; left++) {
      vector<int>& adj_box_indices = boxes_table_[left];
      for (int i = 0; i < adj_box_indices.size(); i++) {
        int adj_box_index = adj_box_indices[i];
        if (meet_v_iou(adj_box_index, index)) {
          results.push_back(adj_box_index);
        }
      }
      if (results.size() != 0) {
        return results;
      }
    }
    return results;
  }

  vector<int> get_precursors(int index) {
    vector<float> box = text_proposals_[index];
    vector<int> results;
    int max_index = std::max(int(box[0] - kMAX_HORIZONTAL_GAP), 0);
    for (int left = int(box[0]) - 1; left > max_index - 1; left--) {
      vector<int>& adj_box_indices = boxes_table_[left];
      for (int i = 0; i < adj_box_indices.size(); i++) {
        int adj_box_index = adj_box_indices[i];
        if (meet_v_iou(adj_box_index, index)) {
          results.push_back(adj_box_index);
        }
      }
      if (results.size() != 0) {
        return results;
      }
    }
    return results;
  }

  bool is_succession_node(int index, int succession_index) {
    vector<int> precursors = get_precursors(succession_index);
    float max_v = -1.0;
    for (int i = 0; i < precursors.size(); i++) {
      if (scores_[precursors[i]] > max_v) {
        max_v = scores_[precursors[i]];
      }
    }
    if (scores_[index] >= max_v) {
      return true;
    }

    return false;
  }

  bool meet_v_iou(int index1, int index2) {
    return ((overlaps_v(index1, index2) >= kMIN_V_OVERLAPS) &&
            (size_similarity(index1, index2) >= kMIN_SIZE_SIM));
  }

  float overlaps_v(int index1, int index2) {
    float h1 = heights_[index1];
    float h2 = heights_[index2];
    int y0 = std::max(text_proposals_[index2][1], text_proposals_[index1][1]);
    int y1 = std::min(text_proposals_[index2][3], text_proposals_[index1][3]);
    return std::max(0, y1 - y0 + 1) * 1.0 / (std::min(h1, h2) * 1.0);
  }

  float size_similarity(int index1, int index2) {
    float h1 = heights_[index1];
    float h2 = heights_[index2];
    return (std::min(h1, h2)) / (std::max(h1, h2));
  }

  Graph BuildGraph(vector<vector<float>>& text_proposals, vector<float>& scores,
                   vector<int>& im_size) {
    text_proposals_ = text_proposals;
    scores_ = scores;
    im_size_ = im_size;
    for (int i = 0; i < text_proposals_.size(); i++) {
      heights_.push_back(text_proposals_[i][3] - text_proposals_[i][1] + 1);
    }

    vector<vector<int>> boxes_table(im_size_[1]);
    for (int index = 0; index < text_proposals_.size(); index++) {
      boxes_table[int(text_proposals_[index][0])].push_back(index);
    }
    boxes_table_ = boxes_table;

    graph_.resize(text_proposals_.size());
    for (int i = 0; i < text_proposals_.size(); i++) {
      graph_[i].resize(text_proposals_.size());
    }

    for (int index = 0; index < text_proposals_.size(); index++) {
      vector<int> successions = get_successions(index);
      if (successions.size() == 0) {
        continue;
      }

      int succession_index = 0;
      float score_max = -1.0;
      for (int i = 0; i < successions.size(); i++) {
        if (scores_[successions[i]] > score_max) {
          score_max = scores_[successions[i]];
          succession_index = successions[i];
        }
      }
      if (is_succession_node(index, succession_index)) {
        graph_[index][succession_index] = true;
      }
    }

    return Graph(graph_);
  }

  vector<vector<float>> text_proposals_;
  vector<float> scores_;
  vector<int> im_size_;
  vector<float> heights_;
  vector<vector<int>> boxes_table_;
  vector<vector<bool>> graph_;
};

}  // namespace jdcn

#endif  // GRAPH_HPP_