#pragma once
#include <sqlite3.h>
#include "common.h"
#include "FaceDatabase.h"  

class OpencvFaceDatabase : public FaceDatabase {
public: 

    OpencvFaceDatabase(const std::string& db_path);
    ~OpencvFaceDatabase();

    // 初始化表结构
    bool init_table() override;

    // 存储人脸数据
    int64_t insert(const Facedata& face, const std::string& img_path) override;

    // 查询数据库人脸数量
    int64_t get_face_count() override;

    // 根据名称查找人脸数据
    std::vector<Facedata> find_by_name(const std::string& name) override;

    // 根据ID查找人脸数据
    std::vector<Facedata> find_by_id(int id) override;

    // 获取所有已知人脸数据（用于程序启动时加载到内存）
    std::vector<Facedata> load_all_faces() override;

    // 根据名称删除人脸数据
    int64_t delete_by_name(const std::string& name) override;

    // 根据ID删除人脸数据
    int64_t delete_by_id(int id) override;

private:
    sqlite3* db_;
    std::string databastpath_;
    mutable std::mutex dbMutex_;
};