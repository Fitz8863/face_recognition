#pragma once
#include <sqlite3.h>
#include <string>
#include <vector>
#include "FaceCode.h"

class FaceDatabase {
public:
    FaceDatabase(const std::string& db_path);
    ~FaceDatabase();

    // 初始化表结构
    bool init_table();

    // 存储人脸数据
    bool insert(const FaceData& face, const std::string& img_path);

    // 查询数据库人脸数量
    int get_face_count();

    // 根据名称查找人脸数据
    std::vector<FaceData> find_by_name(const std::string& name);

    // 根据ID查找人脸数据
    std::vector<FaceData> find_by_id(int id);

    // 获取所有已知人脸数据（用于程序启动时加载到内存）
    std::vector<FaceData> load_all_faces();

    // 根据名称删除人脸数据
    bool delete_by_name(const std::string& name);

    // 根据ID删除人脸数据
    bool delete_by_id(int id);

private:
    sqlite3* db;
    std::string path;

    // 内部辅助函数：将 matrix 转换为二进制 BLOB 存储，或反之
    // 因为 dlib 的 matrix<float,0,1> 是 128 个连续的 float
};