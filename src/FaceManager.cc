#include "FaceManager.h"

FaceManager::FaceManager(const FaceDatabase& database, const FaceCoder& recognizer)
    : db(database), recognizer(recognizer) {
    this->facedatabase = this->db.load_all_faces();
}

// 查询数据库人脸数据数量
int FaceManager::get_databaseface_count() {
    return this->db.get_face_count();
}

// 查询内存中人脸数据数量
int FaceManager::get_bufferface_count() {
    return static_cast<int>(this->facedatabase.size());
}

// 加载所有人脸数据
std::vector<FaceData> FaceManager::load_all_faces() {
    return this->db.load_all_faces();
}

// 注册人脸, 图像路径版本
bool FaceManager::register_face(const std::string& img_path, const std::string& name) {
    auto face = this->recognizer.get_face_data(img_path, name);

    // 检测到的人脸数量判断
    if (face.empty()) {
        std::cout << "未检测到人脸，注册失败。" << std::endl;
        return false;
    }

    else if (face.size() > 1) {
        std::cout << "检测到多张人脸，请确保图像中只有一张人脸，注册失败。" << std::endl;
        return false;
    }

    // 判断这个人脸的名字是否存在
    auto existing_faces = this->db.find_by_name(name);
    // 如果这个人脸的名字，先进行配对一下
    if (!existing_faces.empty()) {
        bool exist = this->recognizer.compare_faces(existing_faces, face);
        if (exist) {
            std::cout << "此人脸已经是存在的了，注册失败。" << std::endl;
            return false;
        }
        // 这种就是出现了同名字的情况了，但是不同一个人,可以对名字进行一些加编号处理
        else {
            face[0].name = name + std::to_string(existing_faces.size());
        }
    }

    // 插入数据库
    bool flag = this->db.insert(face[0], img_path);

    // 同步内存中的数据
    if (flag) {
        this->facedatabase.push_back(face[0]);
    }
    else {
        std::cout << "人脸数据插入数据库失败，注册失败。" << std::endl;
    }

    return flag;
}

// 注册人脸，cv::Mat 版本
bool FaceManager::register_face(const cv::Mat& img, const std::string& name) {
    auto face = this->recognizer.get_face_data(img, name);

    // 检测到的人脸数量判断
    if (face.empty()) {
        std::cout << "未检测到人脸，注册失败。" << std::endl;
        return false;
    }
    else if (face.size() > 1) {
        std::cout << "检测到多张人脸，请确保图像中只有一张人脸，注册失败。" << std::endl;
        return false;
    }

    // 判断这个人脸的名字是否存在
    auto existing_faces = this->db.find_by_name(name);
    // 如果这个人脸的名字，先进行配对一下
    if (!existing_faces.empty()) {
        bool exist = this->recognizer.compare_faces(existing_faces, face);
        if (exist) {
            std::cout << "此人脸已经是存在的了，注册失败。" << std::endl;
            return false;
        }
        // 这种就是出现了同名字的情况了，但是不同一个人,可以对名字进行一些加编号处理
        else {
            face[0].name = name + std::to_string(existing_faces.size());
        }
    }

    // 插入数据库
    bool flag = this->db.insert(face[0], "");

    // 同步内存中的数据
    if (flag) {
        this->facedatabase.push_back(face[0]);
    }

    return flag;
}

// 搜索匹配人脸 (cv::Mat 版本),假如数据库找到匹配人脸则返回 true
bool FaceManager::compare_face(const cv::Mat& img, std::vector<FaceData>& matched_faces, double tolerance) {
    auto target_faces = this->recognizer.get_face_data(img);

    this->recognizer.compare_faces(this->facedatabase, target_faces, tolerance);

    // 收集匹配结果
    for (const auto& face : target_faces) {
        matched_faces.push_back(face);
    }

    return !matched_faces.empty();
}

// 搜索匹配人脸 (图像路径版本)
bool FaceManager::compare_face(const std::string& img_path, std::vector<FaceData>& matched_faces, double tolerance) {
    auto target_faces = this->recognizer.get_face_data(img_path);

    this->recognizer.compare_faces(this->facedatabase, target_faces, tolerance);

    // 收集匹配结果
    for (const auto& face : target_faces) {
        if (face.name != "unknown") {
            matched_faces.push_back(face);
        }
    }

    return !matched_faces.empty();
}

// 查找人脸数据
std::vector<FaceData> FaceManager::find_face_by_name(const std::string& name) {
    return this->db.find_by_name(name);
}

std::vector<FaceData> FaceManager::find_face_by_id(int id) {
    return this->db.find_by_id(id);
}

// 删除人脸数据
bool FaceManager::delete_face_by_name(const std::string& name) {
    bool flag = this->db.delete_by_name(name);

    if (flag) {
        // 同步内存中的数据
        this->facedatabase.erase(
            std::remove_if(
                this->facedatabase.begin(),
                this->facedatabase.end(),
                [&name](const FaceData& fd) { return fd.name == name; }
            ),
            this->facedatabase.end()
        );
    }
    return flag;
}

bool FaceManager::delete_face_by_id(int id) {
    bool flag = this->db.delete_by_id(id);
    if (flag) {
        // 同步内存中的数据
        this->facedatabase.erase(
            std::remove_if(
                this->facedatabase.begin(),
                this->facedatabase.end(),
                [&id](const FaceData& fd) { return fd.id == id; }
            ),
            this->facedatabase.end()
        );
    }
    return flag;
}


// 绘制人脸框
void FaceManager::draw_rectangle(cv::Mat& img, const std::vector<FaceData>& faces) {
    this->recognizer.DrawRectangle(img, faces);
}

FaceManager::~FaceManager() {
}
