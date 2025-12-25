#include "OpencvFaceDatabase.h"


OpencvFaceDatabase::OpencvFaceDatabase(const std::string& db_path) : databastpath_(db_path), db_(nullptr) {
    if (sqlite3_open(this->databastpath_.c_str(), &this->db_) != SQLITE_OK) {
        LOGE("无法打开数据库: " << sqlite3_errmsg(this->db_));
    }
    else {
        if (!init_table()) {
            LOGE("初始化表结构失败。");
        }
    }
}

OpencvFaceDatabase::~OpencvFaceDatabase() {
    if (this->db_) sqlite3_close(this->db_);
}

// 初始化表结构
bool OpencvFaceDatabase::init_table() {

    const char* sql = "CREATE TABLE IF NOT EXISTS opencv_faces ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "user_name TEXT NOT NULL,"
        "img_path TEXT NOT NULL,"
        "face_encoding BLOB NOT NULL,"
        "created_time DATETIME DEFAULT CUR);";

    char* err_msg = nullptr;
    if (sqlite3_exec(this->db_, sql, nullptr, nullptr, &err_msg) != SQLITE_OK) {
        LOGE("创建表失败: " << err_msg);
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

// 查询数据库人脸数量
int OpencvFaceDatabase::get_face_count() {
    const char* sql = "SELECT COUNT(*) FROM opencv_faces;";
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
bool OpencvFaceDatabase::insert(const Facedata& face, const std::string& img_path) {
    if (face.embedding.empty()) {
        LOGE("特征向量为空，拒绝插入数据库");
        return false;
    }
    const char* sql = "INSERT INTO opencv_faces (user_name, img_path ,face_encoding) VALUES (?,?,?);";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(this->db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return false;

    // 1. 绑定姓名
    sqlite3_bind_text(stmt, 1, face.name.c_str(), -1, SQLITE_STATIC);

    // 绑定图片路径
    sqlite3_bind_text(stmt, 2, img_path.c_str(), -1, SQLITE_STATIC);
    // 2. 绑定特征向量 (BLOB)
    int dataSize = face.embedding.size() * sizeof(float); // 对于128维，结果是 512
    // face.encoding(0,0) 获取第一个元素的指针，128 * sizeof(float) 是总字节数 (512字节)
    sqlite3_bind_blob(stmt, 3, face.embedding.data(), dataSize, SQLITE_STATIC);

    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    if (!success) {
        LOGE("插入失败: " << sqlite3_errmsg(this->db_));
    }
    sqlite3_finalize(stmt);

    return success;
}

std::vector<Facedata> OpencvFaceDatabase::load_all_faces() {
    std::vector<Facedata> results;
    const char* sql = "SELECT * FROM opencv_faces;";
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

std::vector<Facedata> OpencvFaceDatabase::find_by_name(const std::string& name) {
    std::vector<Facedata>   results;
    const char* sql = "SELECT * FROM opencv_faces WHERE user_name = ?;";
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
std::vector<Facedata> OpencvFaceDatabase::find_by_id(int id) {
    std::vector<Facedata> results;
    const char* sql = "SELECT id, user_name,img_path ,face_encoding FROM opencv_faces WHERE id = ?;";
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



bool OpencvFaceDatabase::delete_by_name(const std::string& name) {
    const char* sql = "DELETE FROM opencv_faces WHERE user_name = ?;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(this->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);
    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);

    return success;
}

bool OpencvFaceDatabase::delete_by_id(int id) {
    const char* sql = "DELETE FROM opencv_faces WHERE id = ?;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(this->db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
    sqlite3_bind_int(stmt, 1, id);
    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);

    return success;
}