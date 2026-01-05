#pragma once
#include "database/FaceDatabase.h"
#include "DlibFaceCoder.h"
#include <unordered_map>


class DlibRecognizer : public FaceRecognizer {
public:
    DlibRecognizer(const std::string& dbPath,
        const std::string& detectorPath,
        const std::string& recognizerPath);

    // 加载人脸数据
    std::vector<Facedata> load_all_faces();

    // 获取人脸库数量
    int getFacedatabaseCount() override;

    // 在人脸库中注册新的人脸
    bool registerFace(const cv::Mat& image, const std::string& name) override;
    bool registerFace(const std::string path, const std::string& name) override;

    // 在人脸库查找此人脸特征，返回对应人脸结构体
    std::vector<Facedata> recognizeFace(const cv::Mat& faceImage) override;
    
    // 通过name查找人脸库
    std::vector<Facedata> findByNname(const std::string& name) override;

    // 删除人脸数据
    bool deleteFaceByName(const std::string& name) override;

    // 设置阈值（用于判断是否为同一人）
    bool setThreshold(double threshold) override;

    // 绘制人脸框
    void drawFaceBoxes(cv::Mat& image, std::vector<Facedata>& facedata) override;

    // 获取当前使用的后端名称（用于日志/调试）
    std::string getBackendName() const override;
    ~DlibRecognizer();

private:
    std::unique_ptr<FaceDatabase> facedatabase_; // 人脸数据库实例
    std::unique_ptr<DlibFaceCoder> facecoder_;       // 人脸编码器实例

    // 内存中的人脸库（哈希表）
    std::unordered_map<int64_t, Facedata> facedata_map_;

    index_dense_t index_;
    double tolerance_ = TOLERANCE; // 欧氏距离阈值
};