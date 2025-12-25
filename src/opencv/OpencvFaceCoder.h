#pragma once    
#include "config.h"

class OpencvFaceCoder {
public:
    OpencvFaceCoder(const std::string& detectorPath,
        const std::string& recognizerPath);

    // 工厂方法
    static std::unique_ptr<OpencvFaceCoder> create(const std::string& detectorPath,
        const std::string& recognizerPath);

    // 加载模型
    bool loadModels(const std::string& detectorPath,
        const std::string& recognizerPath);

    // 图像预处理
    cv::Mat preprocessImage(const cv::Mat& image);

    // 人脸检测，返回mat格式的人脸框信息
    cv::Mat detectFaces(const cv::Mat& image);

    // 人脸特征提取, 一个图片可能有多个人脸,返回Facedata数组
    std::vector<Facedata> get_facedatas(const cv::Mat& image);

    // 两个人脸特征进行比较, 返回相似度分数
    double compareFeatures(const Facedata& face1, const Facedata& face2);

    // 在图像上绘制人脸框
    void drawFaceBoxes(cv::Mat& image, std::vector<Facedata>& facedata);

private:
    // 人脸检测器和识别器 
    cv::Ptr<cv::FaceDetectorYN> detector_;
    cv::Ptr<cv::FaceRecognizerSF> recognizer_;

    // 选择的后端和目标设备
    int backendId_ = BACKEND_ID;
    int targetId = TARGET_ID;

    double scale_ = 1.0;


    // 类型转换 , cv::Mat 转 std::vector<float>
    std::vector<float> Mat2Vector(const cv::Mat& mat);
    // std::vector<float> 转 cv::Mat
    cv::Mat Vector2Mat(const std::vector<float>& vec);
};