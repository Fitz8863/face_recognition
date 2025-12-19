#pragma once

#include "common.h"



// 人脸编码器类
class FaceCoder
{
public:
    FaceCoder(const std::string &yunet_model_path, const std::string &shape_predictor_path,
              const std::string &face_recognition_model_path);

    // 从图像文件获取所有人脸数据
    std::vector<FaceData> get_face_data(const std::string &image_path, const std::string &name = "unknown");
    std::vector<FaceData> get_face_data(const cv::Mat &cv_img, const std::string &name = "unknown");

    // 人脸检测，返回人脸的方框列表
    std::vector<dlib::full_object_detection> detect_faces(const cv_image<rgb_pixel> &img);

    // 人脸数据结构化
    std::vector<FaceData> img_to_facedata(const cv_image<rgb_pixel> &img, const std::string &name = "unknown");

    // 核心预处理函数
    matrix<rgb_pixel> preprocess_image(const matrix<rgb_pixel> &img);
    // OpenCV图像预处理（重载版本，接收 cv::Mat）
    cv::Mat preprocess_image(const cv::Mat &cv_img);

    // 比较人脸编码,已有的人脸库对比单个人脸编码
    bool compare_faces(const std::vector<FaceData> &known_facedatas,
                       std::vector<FaceData> &face_datas_to_check,
                       double tolerance = TOLERANCE);

    // 绘制人脸框,opencv格式
    void DrawRectangle(cv::Mat &img, std::vector<FaceData> face_datas);

    ~FaceCoder();

private:
    double scale = 1.0; // 实际使用的缩放比例
    int width = 0; // 实际的缩放宽度
    int height = 0; // 实际的缩放高度

    cv::Ptr<cv::FaceDetectorYN> yunet_detector; // YuNet 检测器

    frontal_face_detector detector;
    shape_predictor sp;
    anet_type net;
};