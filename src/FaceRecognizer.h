#pragma once
#include <memory> // for std::unique_ptr if needed
#include "common.h"

// 前向声明 OpenCV 类型（避免头文件污染）
namespace cv
{
    class Mat;
}

class FaceRecognizer {
public:
    virtual ~FaceRecognizer() = default;

    // 工厂模式
    static std::unique_ptr<FaceRecognizer> create(Type type);


    // 在人脸库中注册新的人脸
    virtual bool registerFace(const cv::Mat& image, const std::string& name) = 0;
    virtual bool registerFace(const std::string path, const std::string& name) = 0;

    // 在人脸库查找此人脸特征，返回对应人脸结构体列表
    virtual std::vector<Facedata> recognizeFace(const cv::Mat& faceImage) = 0;

    // 根据name查找数据库人脸数据
    virtual std::vector<Facedata> findByNname(const std::string& name) = 0;

    // 绘制人脸框
    virtual void drawFaceBoxes(cv::Mat& image, std::vector<Facedata>& facedata) = 0;

    // 删除人脸操作
    virtual bool deleteFaceByName(const std::string& name) = 0;

    // 查看人脸库人脸数量
    virtual int getFacedatabaseCount() = 0;

    // 设置阈值（用于判断是否为同一人）
    virtual bool setThreshold(double threshold) = 0;

    // 获取当前使用的后端名称（用于日志/调试）
    virtual std::string getBackendName() const = 0;
};

