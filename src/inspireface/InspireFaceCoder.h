#pragma once
#include "config.h"
#include <inspireface/inspireface.hpp>


class InspireFaceCoder
{
public:
    InspireFaceCoder(const std::string& model_path);

    // 工厂方法
    static std::unique_ptr<InspireFaceCoder> create(const std::string& model_path);

    // 人脸检测，返回人脸框信息
    std::vector<inspire::FaceTrackWrap> detectFaces(const cv::Mat& image);

    // 人脸特征提取, 一个图片可能有多个人脸,返回Facedata数组
    std::vector<Facedata> get_facedatas(const cv::Mat& image);

    // 两个人脸对比，返回余弦相似度
    float compareFeatures(const Facedata& face1, const Facedata& face2);

    // 在图像上绘制人脸框
    void drawFaceBoxes(cv::Mat& image, std::vector<Facedata>& facedata);

    // --------------------------------pipeline--------------------------------------------------------------------------
    //  pipeline 初始化
    void pipeline_init(const std::vector<inspire::FaceTrackWrap>& results, inspirecv::FrameProcess& process);

    // 返回人脸状态检测结果
    std::vector<FaceStateInfo> StateDetect(const cv::Mat& image);

    // 人脸 RGB 防欺骗，返回检测人脸的置信度数组
    std::vector<float> rgbLivenessDetect(const cv::Mat& image);

    // 口罩检测,返回图片人脸口罩置信度数组
    std::vector<float> MaskDetect(const cv::Mat& image);

    // 人脸质量预测，返回图片人脸置信度数组
    std::vector<float> QualityDetect(const cv::Mat& image);

    // 人脸属性检测,返回人种，性别，年龄信息
    std::vector<inspire::FaceAttributeResult> AttributeDetect(const cv::Mat& image);

    // 面部表情检测
    std::vector<inspire::FaceEmotionResult> EmotionDetect(const cv::Mat& image);

    // 眼睛状态检测
    std::vector<inspire::FaceInteractionState> EyeStateDetect(const cv::Mat& image);

    // 面部交互动作检测
    std::vector<inspire::FaceInteractionAction> InteractionActionDetect(const cv::Mat& image);

    ~InspireFaceCoder();

private:


    inspire::CustomPipelineParameter param_;
    // inspirecv::FrameProcess process_;
    std::unique_ptr<inspire::Session> session_;

    double scale_ = 1.0;
    mutable std::mutex faceMutex_;
};
