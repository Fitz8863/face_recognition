#pragma once
#include <sqlite3.h>
#include "common.h"

class FaceDatabase {
public:
    // 工厂方法
    static std::unique_ptr<FaceDatabase> create(const std::string& db_path,Type type);

    // 初始化表结构
    virtual bool init_table() = 0;

    // 插入操作
    virtual int64_t insert(const Facedata& face, const std::string& img_path) = 0;

    // 查询数据库人脸数量
    virtual int64_t get_face_count() = 0;

    // 根据名称查找人脸数据
    virtual std::vector<Facedata> find_by_name(const std::string& name) = 0;

    // 根据ID查找人脸数据
    virtual std::vector<Facedata> find_by_id(int id) = 0;

    // 获取所有已知人脸数据（用于程序启动时加载到内存）
    virtual std::vector<Facedata> load_all_faces() = 0;

    // 根据名称删除人脸数据
    virtual int64_t delete_by_name(const std::string& name) = 0;

    // 根据ID删除人脸数据
    virtual int64_t delete_by_id(int id) = 0;

private:
    sqlite3* db_;
    std::string databastpath_;
    Type type_;
};