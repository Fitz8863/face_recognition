#pragma once
#include "FaceDatabase.h"

class FaceManager {
public:
    FaceManager(const FaceDatabase& database, const FaceCoder& recognizer);

    // 加载人脸数据
    std::vector<FaceData> load_all_faces();

    // 查询数据库人脸数据数量
    int get_databaseface_count();
    
    // 查询内存中人脸数据数量
    int get_bufferface_count();

    // 添加人脸数据到数据库,注册人脸
    bool register_face(const std::string& img_path, const std::string& name);
    bool register_face(const cv::Mat& img, const std::string& name);

    // 查找人脸库
    std::vector<FaceData> find_face_by_name(const std::string& name);
    std::vector<FaceData> find_face_by_id(int id);

    // 删除人脸数据
    bool delete_face_by_name(const std::string& name);
    bool delete_face_by_id(int id);

    // 匹配人脸
    bool compare_face(const cv::Mat& img, std::vector<FaceData>& matched_faces, double tolerance = TOLERANCE);
    bool compare_face(const std::string& img_path, std::vector<FaceData>& matched_faces, double tolerance = TOLERANCE);

    // 绘制人脸框
    void draw_rectangle(cv::Mat& img, const std::vector<FaceData>& faces);

    ~FaceManager();

private:
    FaceDatabase db;
    FaceCoder recognizer;

    std::vector<FaceData> facedatabase;
};