#pragma once
#include <dlib/opencv.h>
#include "config.h"



// 人脸编码器类
class DlibFaceCoder
{
public:
    DlibFaceCoder(const std::string& shape_predictor_path,
        const std::string& face_recognition_model_path);

    // 工厂方法
    static std::unique_ptr<DlibFaceCoder> create(const std::string& detectorPath,
        const std::string& recognizerPath);

    // 加载模型
    bool loadModels(const std::string& detectorPath,
        const std::string& recognizerPath);

    // 预处理函数
    matrix<rgb_pixel> preprocess_image(const matrix<rgb_pixel>& img);
    // OpenCV图像预处理（重载版本，接收 cv::Mat）
    cv::Mat preprocess_image(const cv::Mat& cv_img);

    // 人脸检测，返回人脸的方框列表
    std::vector<dlib::full_object_detection> detect_faces(const cv_image<rgb_pixel>& img);

    // 从图像文件获取所有人脸数据
    std::vector<Facedata> get_facedatas(const cv::Mat& cv_img);

    // 比较人脸编码,已有的人脸库对比单个人脸编码
    double compareFeatures(const Facedata& face1, const Facedata& face2);

    // 绘制人脸框,opencv格式
    void DrawRectangle(cv::Mat& img, std::vector<Facedata> face_datas);

    ~DlibFaceCoder();

private:
    double scale_ = 1.0; // 实际使用的缩放比例

    frontal_face_detector detector_;
    shape_predictor sp_;
    anet_type net_;


    // 类型转换函数,dlib人脸特征向量转为vector中
    std::vector<float> Matrix2Vector(matrix<float, 0, 1>& encoding);
    // vector转为matrix<float, 0, 1>类型
    dlib::matrix<float, 0, 1>  Vector2Matrix(std::vector<float> vec);
};