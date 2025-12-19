#include "FaceDatabase.h"


FaceDatabase::FaceDatabase(const std::string& db_path) : path(db_path), db(nullptr) {
    if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) {
        std::cerr << "无法打开数据库: " << sqlite3_errmsg(db) << std::endl;
    }
    else {
        if (!init_table()) {
            std::cerr << "初始化表结构失败。" << std::endl;
        }
    }
}

FaceDatabase::~FaceDatabase() {
    if (db) sqlite3_close(db);
}

// 初始化表结构
bool FaceDatabase::init_table() {
    const char* sql = "CREATE TABLE IF NOT EXISTS faces ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "user_name TEXT NOT NULL,"
        "img_path TEXT NOT NULL,"
        "face_encoding BLOB NOT NULL,"
        "created_time DATETIME DEFAULT CURRENT_TIMESTAMP);";

    char* err_msg = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err_msg) != SQLITE_OK) {
        std::cerr << "❌ 创建表失败: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

// 查询数据库人脸数量
int FaceDatabase::get_face_count() {
    const char* sql = "SELECT COUNT(*) FROM faces;";
    sqlite3_stmt* stmt;
    int count = 0;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return count;

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }

    sqlite3_finalize(stmt);
    return count;
}

// 插入操作
bool FaceDatabase::insert(const FaceData& face, const std::string& img_path) {
    const char* sql = "INSERT INTO faces (user_name, img_path ,face_encoding) VALUES (?,?,?);";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return false;

    // 1. 绑定姓名
    sqlite3_bind_text(stmt, 1, face.name.c_str(), -1, SQLITE_STATIC);

    // 绑定图片路径
    sqlite3_bind_text(stmt, 2, img_path.c_str(), -1, SQLITE_STATIC);
    // 2. 绑定特征向量 (BLOB)
    // face.encoding(0,0) 获取第一个元素的指针，128 * sizeof(float) 是总字节数 (512字节)
    sqlite3_bind_blob(stmt, 3, &face.encoding(0, 0), 128 * sizeof(float), SQLITE_STATIC);

    
    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);

    return success;
}

std::vector<FaceData> FaceDatabase::load_all_faces() {
    std::vector<FaceData> results;
    const char* sql = "SELECT * FROM faces;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        FaceData fd;
        fd.id = sqlite3_column_int(stmt, 0);
        fd.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        // 从 BLOB 读取并恢复 matrix
        const void* blob_data = sqlite3_column_blob(stmt, 3);
        int bytes = sqlite3_column_bytes(stmt, 3);

        if (bytes == 128 * sizeof(float)) {
            // 直接拷贝内存数据到 dlib matrix 的底层指针处
            fd.encoding.set_size(128, 1);
            memcpy(&fd.encoding(0, 0), blob_data, bytes);
            results.push_back(fd);
        }
    }

    sqlite3_finalize(stmt);
    return results;
}

std::vector<FaceData> FaceDatabase::find_by_name(const std::string& name) {
    std::vector<FaceData>   results;
    const char* sql = "SELECT * FROM faces WHERE user_name = ?;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        FaceData fd;
        fd.id = sqlite3_column_int(stmt, 0);
        fd.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        // 从 BLOB 读取并恢复 matrix
        const void* blob_data = sqlite3_column_blob(stmt, 3);
        int bytes = sqlite3_column_bytes(stmt, 3);

        if (bytes == 128 * sizeof(float)) {
            fd.encoding.set_size(128, 1);
            memcpy(&fd.encoding(0, 0), blob_data, bytes);
            results.push_back(fd);
        }
    }

    sqlite3_finalize(stmt);
    return results;
}



// 数据库通过id号查找人脸数据
std::vector<FaceData> FaceDatabase::find_by_id(int id) {
    std::vector<FaceData> results;
    const char* sql = "SELECT id, user_name,img_path ,face_encoding FROM faces WHERE id = ?;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

    sqlite3_bind_int(stmt, 1, id);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        FaceData fd;
        fd.id = sqlite3_column_int(stmt, 0);
        fd.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        // 从 BLOB 读取并恢复 matrix
        const void* blob_data = sqlite3_column_blob(stmt, 3);
        int bytes = sqlite3_column_bytes(stmt, 3);

        if (bytes == 128 * sizeof(float)) {
            fd.encoding.set_size(128, 1);
            memcpy(&fd.encoding(0, 0), blob_data, bytes);
            results.push_back(fd);
        }
    }

    sqlite3_finalize(stmt);
    return results;
}



bool FaceDatabase::delete_by_name(const std::string& name) {
    const char* sql = "DELETE FROM faces WHERE user_name = ?;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);
    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);

    return success;
}

bool FaceDatabase::delete_by_id(int id) {
    const char* sql = "DELETE FROM faces WHERE id = ?;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
    sqlite3_bind_int(stmt, 1, id);
    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);

    return success;
}