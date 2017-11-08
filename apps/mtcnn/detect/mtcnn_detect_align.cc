#include "mtcnn_detect_align.h"

MTCNNDetectAlign::MTCNNDetectAlign(const string& model_dir) {
    Init(model_dir);
}

bool MTCNNDetectAlign::Init(const string& model_dir) {
    mtcnn.reset(new MTCNN(model_dir));

    string lbf_regress_model = model_dir + "/LBF.model";
    lbf_cascador.reset(new LbfCascador());
    FILE *fd = fopen(lbf_regress_model.c_str(), "rb");
    lbf_cascador->Read(fd);
    fclose(fd);
    return true;
}

int MTCNNDetectAlign::Detect(const cv::Mat& image, vector<JDFaceInfo>& face_infos) {

    std::vector<FaceInfo> faceInfo;
    mtcnn->Detect(image, faceInfo);
    
    cv::Mat gray;
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    for(int i = 0;i<faceInfo.size();i++){
        float x = faceInfo[i].bbox.x1;
        float y = faceInfo[i].bbox.y1;
        float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 +1;
        float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 +1;
        JDFaceInfo jd_face_info;
        jd_face_info.face_bbox = cv::Rect(y, x, w, h);
        jd_face_info.score = faceInfo[i].bbox.score;

        BBox facebox(std::max<int>(0, y), std::max<int>(0, x), w, h);
        Mat face_shape = lbf_cascador->Predict(gray, facebox);
        vector<cv::Point> face_points;
        for (int ipt = 0; ipt < 68; ipt++)
        {
            int xpt = face_shape.at<double>(ipt, 0);
            int ypt = face_shape.at<double>(ipt, 1);
            face_points.push_back(cv::Point(xpt, ypt));
        }

        jd_face_info.face_points = face_points;
        face_infos.push_back(jd_face_info);
    }

}