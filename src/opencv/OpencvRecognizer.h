#pragma once
#include "FaceRecognizer.h"
#include "database/FaceDatabase.h"
#include "OpencvFaceCoder.h"
#include <unordered_map>

class OpencvRecognizer : public FaceRecognizer
{
public:
    OpencvRecognizer(const std::string &dbPath,
                     const std::string &detectorPath,
                     const std::string &recognizerPath);

    // 在人脸库中注册新的人脸
    bool registerFace(const cv::Mat &image, const std::string &name) override;
    bool registerFace(const std::string path, const std::string &name) override;

    // 在人脸库查找此人脸特征，返回对应人脸结构体
    std::vector<Facedata> recognizeFace(const cv::Mat &faceImage) override;
    // 通过name查找人脸库
    std::vector<Facedata> findByNname(const std::string &name) override;

    // 删除人脸操作
    bool deleteFaceByName(const std::string &name) override;

    // 获取人脸库数量
    int getFacedatabaseCount() override;

    // 绘制人脸框
    void drawFaceBoxes(cv::Mat &image, std::vector<Facedata> &facedata) override;

    // 设置阈值（用于判断是否为同一人）
    bool setThreshold(double threshold) override;

    // 获取当前使用的后端名称（用于日志/调试）
    std::string getBackendName() const override;

    ~OpencvRecognizer() override;

private:
    std::unique_ptr<FaceDatabase> facedatabase_; // 人脸数据库实例
    std::unique_ptr<OpencvFaceCoder> facecoder_; // 人脸编码器实例

    // 人脸数据全部加载到内存
    std::unordered_map<uint64_t, Facedata> facedata_map_;

    index_dense_t index_;

    double threshold_ = RECOGNIZER_CONFIDENCE_THRESHOLD; // 相似度阈值
};
