#define SIMPLE_EXPORT
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "graph.hpp"
#include "ployfit.hpp"

using namespace cv;
using namespace caffe;
using namespace std;

void clipBoxes(vector<float>& box, int height, int width) {
	if (box[0] < 0) {
		box[0] = 0;
	}
	if (box[0] > width - 1) {
		box[0] = width - 1;
	}
	if (box[2] < 0) {
		box[2] = 0;
	}
	if (box[2] > width - 1) {
		box[2] = width - 1;
	}
	if (box[1] < 0) {
		box[1] = 0;
	}
	if (box[1] > height - 1) {
		box[1] = height - 1;
	}
	if (box[3] < 0) {
		box[3] = 0;
	}
	if (box[3] > height - 1) {
		box[3] = height - 1;
	}
}

int main(int argc, char* argv) {
	Caffe::set_mode(Caffe::CPU);
	string model_file = "D:\\BaiduNetdiskDownload\\ctpn/deploy.prototxt";
	string trained_file =
		"D:\\BaiduNetdiskDownload\\ctpn/ctpn_trained_model.caffemodel";
	string image_name = "D:\\tests/5.jpg";
	boost::shared_ptr<Net<float>> net(new Net<float>(model_file, TEST));

	net->CopyTrainedLayersFrom(trained_file);

	boost::shared_ptr<Blob<float>> blob_data = net->blob_by_name("data");
	boost::shared_ptr<Blob<float>> im_info = net->blob_by_name("im_info");
	// 102.9801, 115.9465, 122.7717

	cv::Mat im = cv::imread(image_name);
	cv::Mat image;
	im.convertTo(image, CV_32FC3);

	image -= cv::Scalar(102.9801, 115.9465, 122.7717);
	int width = image.cols;
	int height = image.rows;
	blob_data->Reshape(1, 3, height, width);
	float* blob_data_ptr = blob_data->mutable_cpu_data();
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			blob_data_ptr[(0 * height + h) * width + w] =
				float(image.at<cv::Vec3f>(cv::Point(w, h))[0]);
			blob_data_ptr[(1 * height + h) * width + w] =
				float(image.at<cv::Vec3f>(cv::Point(w, h))[1]);
			blob_data_ptr[(2 * height + h) * width + w] =
				float(image.at<cv::Vec3f>(cv::Point(w, h))[2]);
		}
	}
	float* im_data = im_info->mutable_cpu_data();
	im_data[0] = height;
	im_data[1] = width;
	net->Forward();
	boost::shared_ptr<Blob<float>> rois = net->blob_by_name("rois");
	boost::shared_ptr<Blob<float>> scores = net->blob_by_name("scores");
	float min_score = 0.7;
	float* scores_data = scores->mutable_cpu_data();
	float* rois_data = rois->mutable_cpu_data();
	vector<vector<float> > text_proposals;
	vector<float> vec_scores;
	for (int i = 0; i < scores->shape()[0]; i++) {
		if (scores_data[i] > min_score) {
			vector<float> vec_roi;
			vec_roi.push_back(rois_data[i * 4 + 0]);
			vec_roi.push_back(rois_data[i * 4 + 1]);
			vec_roi.push_back(rois_data[i * 4 + 2]);
			vec_roi.push_back(rois_data[i * 4 + 3]);
			clipBoxes(vec_roi, height, width);
			text_proposals.push_back(vec_roi);
			vec_scores.push_back(scores_data[i]);
		}
	}

	vector<int> im_size(2);
	im_size[0] = height;
	im_size[1] = width;
	TextProposalGraphBuilder builder;
	Graph g = builder.build_graph(text_proposals, vec_scores, im_size);
	vector<vector<int> > tp_groups = g.sub_graphs_connected();

	cv::Mat show_image = im.clone();

	vector<vector<float> > text_lines;
	for (int i = 0; i < tp_groups.size(); i++) {
		vector<int> tp_group = tp_groups[i];

		vector<vector<float> > text_line_boxes;

		float x0 = 100000.0;
		float x1 = 0.0;
		vector<float> lt_x0;
		vector<float> rt_y0;
		vector<float> lb_x0;
		vector<float> rb_y0;
		for (int j = 0; j < tp_group.size(); j++) {
			int index = tp_group[j];
			text_line_boxes.push_back(text_proposals[index]);
			if (text_proposals[index][0] < x0) {
				x0 = text_proposals[index][0];
			}
			if (text_proposals[index][2] > x1) {
				x1 = text_proposals[index][2];
			}
			lt_x0.push_back(text_proposals[index][0]);
			rt_y0.push_back(text_proposals[index][1]);

			lb_x0.push_back(text_proposals[index][0]);
			rb_y0.push_back(text_proposals[index][3]);
		}
		float offset = (text_line_boxes[0][2] - text_line_boxes[0][0])*0.5;
		czy::Fit fitline;
		fitline.polyfit(lt_x0, rt_y0, 1);
		vector<double> factor;
		fitline.getFactor(factor);
		float lt_y = factor[1] * (x0 + offset) + factor[0];
		float rt_y = factor[1] * (x1 - offset) + factor[0];

		fitline.polyfit(lb_x0, rb_y0, 1);
		fitline.getFactor(factor);
		float lb_y = factor[1] * (x0 + offset) + factor[0];
		float rb_y = factor[1] * (x1 - offset) + factor[0];

		vector<float> text_line(5);
		text_line[0] = (x0);
		text_line[1] = std::min((lt_y), (rt_y));
		text_line[2] = (x1);
		text_line[3] = std::max((lb_y), (rb_y));
		cv::rectangle(show_image, cv::Rect(text_line[0], text_line[1],
			text_line[2] - text_line[0], text_line[3] - text_line[1]), cv::Scalar(0, 255, 0), 2);
		double sum = 0.0;
		for (int j = 0; j < tp_group.size(); j++) {
			sum += vec_scores[tp_group[j]];
		}
		text_line[4] = (sum / tp_group.size());

		text_lines.push_back(text_line);
	}

	cv::imwrite("show.png", show_image);
	return 0;
}