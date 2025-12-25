#include "DlibFaceDatabase.h"

DlibFaceDatabase::DlibFaceDatabase(const std::string& db_path) : databastpath_(db_path), db_(nullptr) {
    if (sqlite3_open(this->databastpath_.c_str(), &this->db_) != SQLITE_OK) {
        std::cerr << "无法打开数据库: " << sqlite3_errmsg(this->db_) << std::endl;
    }
    else {
        if (!init_table()) {
            std::cerr << "初始化表结构失败。" << std::endl;
        }
    }
}

DlibFaceDatabase::~DlibFaceDatabase() {
    if (this->db_) sqlite3_close(this->db_);
}

// 初始化表结构
bool DlibFaceDatabase::init_table() {
    const char* sql = "CREATE TABLE IF NOT EXISTS faces ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "user_name TEXT NOT NULL,"
        "img_path TEXT NOT NULL,"
        "face_encoding BLOB NOT NULL,"
        "created_time DATETIME DEFAULT CURRENT_TIMESTAMP);";

    char* err_msg = nullptr;
    if (sqlite3_exec(this->db_, sql, nullptr, nullptr, &err_msg) != SQLITE_OK) {
        std::cerr << "❌ 创建表失败: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

// 查询数据库人脸数量
int DlibFaceDatabase::get_face_count() {
    const char* sql = "SELECT COUNT(*) FROM faces;";
    sqlite3_stmt* stmt;
    int count = 0;

    if (sqlite3_prepare_v2(this->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return count;

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }

    sqlite3_finalize(stmt);
    return count;
}

// 插入操作
bool DlibFaceDatabase::insert(const Facedata& face, const std::string& img_path) {
    const char* sql = "INSERT INTO faces (user_name, img_path ,face_encoding) VALUES (?,?,?);";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(this->db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return false;

    // 1. 绑定姓名
    sqlite3_bind_text(stmt, 1, face.name.c_str(), -1, SQLITE_STATIC);

    // 绑定图片路径
    sqlite3_bind_text(stmt, 2, img_path.c_str(), -1, SQLITE_STATIC);
    // 2. 绑定特征向量 (BLOB)
    int dataSize = face.embedding.size() * sizeof(float);
    sqlite3_bind_blob(stmt, 3, face.embedding.data(), dataSize, SQLITE_STATIC);

    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);

    return success;
}

std::vector<Facedata> DlibFaceDatabase::load_all_faces() {
    std::vector<Facedata> results;
    const char* sql = "SELECT * FROM faces;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(this->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Facedata fd;
        fd.id = sqlite3_column_int(stmt, 0);
        fd.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        // 从 BLOB 读取并恢复 matrix
        const void* blobPtr = sqlite3_column_blob(stmt, 3);
        int totalBytes = sqlite3_column_bytes(stmt, 3);

        if (blobPtr != nullptr && totalBytes > 0) {
            // 计算 float 的个数
            int elementCount = totalBytes / sizeof(float);

            // 将二进制数据还原为 vector<float>
            float* floatPtr = (float*)blobPtr;
            fd.embedding.assign(floatPtr, floatPtr + elementCount);
            results.push_back(fd);
        }
    }

    sqlite3_finalize(stmt);
    return results;
}

std::vector<Facedata> DlibFaceDatabase::find_by_name(const std::string& name) {
    std::vector<Facedata>   results;
    const char* sql = "SELECT * FROM faces WHERE user_name = ?;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(this->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Facedata fd;
        fd.id = sqlite3_column_int(stmt, 0);
        fd.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        // 从 BLOB 读取并恢复 matrix
        const void* blobPtr = sqlite3_column_blob(stmt, 3);
        int totalBytes = sqlite3_column_bytes(stmt, 3);

        if (blobPtr != nullptr && totalBytes > 0) {
            // 计算 float 的个数
            int elementCount = totalBytes / sizeof(float);

            // 将二进制数据还原为 vector<float>
            float* floatPtr = (float*)blobPtr;
            fd.embedding.assign(floatPtr, floatPtr + elementCount);
            results.push_back(fd);
        }
    }

    sqlite3_finalize(stmt);
    return results;
}



// 数据库通过id号查找人脸数据
std::vector<Facedata> DlibFaceDatabase::find_by_id(int id) {
    std::vector<Facedata> results;
    const char* sql = "SELECT id, user_name,img_path ,face_encoding FROM faces WHERE id = ?;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(this->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

    sqlite3_bind_int(stmt, 1, id);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Facedata fd;
        fd.id = sqlite3_column_int(stmt, 0);
        fd.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        // 从 BLOB 读取并恢复 matrix
        const void* blobPtr = sqlite3_column_blob(stmt, 3);
        int totalBytes = sqlite3_column_bytes(stmt, 3);

        if (blobPtr != nullptr && totalBytes > 0) {
            // 计算 float 的个数
            int elementCount = totalBytes / sizeof(float);

            // 将二进制数据还原为 vector<float>
            float* floatPtr = (float*)blobPtr;
            fd.embedding.assign(floatPtr, floatPtr + elementCount);
            results.push_back(fd);
        }
    }

    sqlite3_finalize(stmt);
    return results;
}



bool DlibFaceDatabase::delete_by_name(const std::string& name) {
    const char* sql = "DELETE FROM faces WHERE user_name = ?;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(this->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);
    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);

    return success;
}

bool DlibFaceDatabase::delete_by_id(int id) {
    const char* sql = "DELETE FROM faces WHERE id = ?;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(this->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
    sqlite3_bind_int(stmt, 1, id);
    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);

    return success;
}