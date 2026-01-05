#pragma once
#include <memory> // for std::unique_ptr if needed
#include "common.h"

// 前向声明 OpenCV 类型（避免头文件污染）
namespace cv
{
    class Mat;
}

class FaceRecognizer
{
public:
    virtual ~FaceRecognizer() = default;

    // 工厂模式
    /**
     * @brief 创建人脸识别器实例
     * @param type 人脸识别器类型
     * @return 返回创建的实例
     */
    static std::unique_ptr<FaceRecognizer> create(Type type);

    /**
     * @brief 在人脸库中注册新的人脸
     * @param image 人脸图片
     * @param name 人脸名称
     * @return 成功返回 true，失败返回 false
     */
    virtual bool registerFace(const cv::Mat &image, const std::string &name) = 0;
    /**
     * @brief 在人脸库中注册新的人脸
     * @param path 人脸图片路径
     * @param name 人脸名称
     * @return 成功返回 true，失败返回 false
     */
    virtual bool registerFace(const std::string path, const std::string &name) = 0;

    /**
     * @brief 在人脸库中查找人脸特征
     * @param faceImage 人脸图片
     * @return 匹配到的人脸结构体列表
     */
    virtual std::vector<Facedata> recognizeFace(const cv::Mat &faceImage) = 0;

    // 根据name查找数据库人脸数据
    /**
     * @brief 根据name查找数据库人脸数据
     * @param name 人脸名称
     * @return 匹配到的人脸结构体列表
     */
    virtual std::vector<Facedata> findByNname(const std::string &name) = 0;

    /**
     * @brief 绘制人脸框
     * @param image 输入图片
     * @param facedata 匹配到的人脸结构体列表
     */
    virtual void drawFaceBoxes(cv::Mat &image, std::vector<Facedata> &facedata) = 0;

    /**
     * @brief 删除人脸操作
     * @param name 人脸名称
     * @return 删除成功返回 true，失败返回 false
     */
    virtual bool deleteFaceByName(const std::string &name) = 0;

    /**
     * @brief 查看人脸库人脸数量
     * @return 人脸数量
     */
    virtual int getFacedatabaseCount() = 0;

    // 设置阈值（用于判断是否为同一人）
    /**
     * @brief 设置阈值
     * @param threshold 阈值
     */
    virtual bool setThreshold(double threshold) = 0;

    // 获取当前使用的后端名称（用于日志/调试）
    /**
     * @brief 获取当前使用的后端名称
     * @return 后端名称
     */
    virtual std::string getBackendName() const = 0;

    /**
     * @brief 活体检测接口
     * @return 成功返回结果对象指针，失败或未检出建议返回 nullptr
     */
    virtual std::unique_ptr<FaceStateInfo> Alivedetect(const cv::Mat &image)
    {
        return nullptr;
    };
};
