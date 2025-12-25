#include "DlibRecognizer.h"

DlibRecognizer::DlibRecognizer(const std::string& dbPath,
    const std::string& detectorPath,
    const std::string& recognizerPath) {

    this->facedatabase_ = FaceDatabase::create(dbPath, DLIB);
    this->facecoder_ = DlibFaceCoder::create(detectorPath, recognizerPath);

    // 加载人脸数据库中的所有人脸数据到内存
    std::vector<Facedata> facedatas = facedatabase_->load_all_faces();
    for (const auto& face : facedatas)
    {
        this->facedata_map_[face.name] = face;
    };

}

// 查询数据库人脸数据数量
int DlibRecognizer::getFacedatabaseCount() {
    return this->facedatabase_->get_face_count();
}

// 注册人脸,cv::Mat 版本
bool DlibRecognizer::registerFace(const cv::Mat& image, const std::string& name) {

    std::string registered_name = name;
    // 检测人脸库里面是否有这个人脸，这里做特征向量匹配
    std::vector<Facedata> list = this->recognizeFace(image);
    for (const auto& face : list) {
        if (face.name != "unknown")
        {
            LOGE("已存在此人脸，请勿重复注册，名字:"<<face.name);
            return false;
        }
    }

    // 提取人脸特征
    std::vector<Facedata> newFaces = facecoder_->get_facedatas(image);
    if (newFaces.empty())
    {
        LOGE("未检测到人脸，注册失败");
        return false;
    }
    else if (newFaces.size() > 1)
    {
        LOGW("检测到多张人脸，注册失败");
        return false;
    }

    // 判断这个人脸的名字是否存在
    auto existing_faces = this->facedatabase_->find_by_name(name);
    int count = existing_faces.size();
    if (count > 0)
    {
        registered_name += std::to_string(count);
        LOGW("名称:" << name << "已存在，注册为新名称: " << registered_name);
    }

    // 开始注册操作
    Facedata& newFace = newFaces[0];
    newFace.name = registered_name;

    // 插入到数据库
    bool success = this->facedatabase_->insert(newFace, ""); // img_path 可选，这里传空字符串
    if (success)
    {
        // 同时添加到内存中的人脸数据
        this->facedata_map_[registered_name] = newFace;
    }
    return success;
}

// 注册人脸， 图像路径版本
bool DlibRecognizer::registerFace(const std::string path, const std::string& name) {

    cv::Mat image = cv::imread(path);
    std::string registered_name = name;
    // 判空
    if (image.empty())
    {
        LOGE("无法读取图像文件: {}" << path);
        return false;
    }

    // 检测人脸库里面是否有这个人脸，这里做特征向量匹配
    std::vector<Facedata> list = this->recognizeFace(image);
    for (const auto& face : list) {
        if (face.name != "unknown")
        {
            LOGE("已存在此人脸，请勿重复注册，名字:"<<face.name);
            return false;
        }
    }

    // 提取人脸特征
    std::vector<Facedata> newFaces = facecoder_->get_facedatas(image);
    if (newFaces.empty())
    {
        LOGE("未检测到人脸，注册失败");
        return false;
    }
    else if (newFaces.size() > 1)
    {
        LOGW("检测到多张人脸，注册失败");
        return false;
    }

    // 判断这个人脸的名字是否存在
    auto existing_faces = this->facedatabase_->find_by_name(name);
    int count = existing_faces.size();
    if (count > 0)
    {
        registered_name += std::to_string(count);
        LOGW("名称:" << name << "已存在，注册为新名称: " << registered_name);
    }

    // 开始注册操作
    Facedata& newFace = newFaces[0];
    newFace.name = registered_name;

    // 插入到数据库
    bool success = this->facedatabase_->insert(newFace, path); // img_path 可选，这里传空字符串
    if (success)
    {
        // 同时添加到内存中的人脸数据
        this->facedata_map_[registered_name] = newFace;
    }
    return success;
}

// 在人脸库查找此人脸特征，返回对应人脸结构体
std::vector<Facedata> DlibRecognizer::recognizeFace(const cv::Mat& faceImage) {
    std::vector<Facedata> queryFaces = this->facecoder_->get_facedatas(faceImage);

    if (queryFaces.empty())
    {
        // LOGE("未检测到人脸，识别失败");
        return queryFaces;
    }

    for (auto& queryFace : queryFaces)
    {
        for (const auto& [_, dbFace] : this->facedata_map_)
        {
            double distance = facecoder_->compareFeatures(queryFace, dbFace);
            // 如果两个人脸的欧氏距离小于阈值，认为是同一人
            if (distance <= this->tolerance_)
            {
                // 更新查询人脸的信息为匹配到的数据库人脸
                queryFace.id = dbFace.id;
                queryFace.name = dbFace.name;
                queryFace.score = distance;
                break;
            }
            else
            {
                // LOGI("识别失败，计算出来的欧氏距离是: " << distance);
            }
        }
    }
    return queryFaces;
}


// 查找人脸数据
std::vector<Facedata> DlibRecognizer::findByNname(const std::string& name) {
    return this->facedatabase_->find_by_name(name);
}


// 删除人脸数据
bool DlibRecognizer::deleteFaceByName(const std::string& name) {
    // 从数据库删除
    bool success = facedatabase_->delete_by_name(name);
    if (success)
    {
        // 同时从内存中删除
        this->facedata_map_.erase(name);
    }
    else
    {
        LOGE("删除人脸失败: {}" << name);
    }
    return success;
}


// 绘制人脸框
void DlibRecognizer::drawFaceBoxes(cv::Mat& image, std::vector<Facedata>& facedata)
{
    this->facecoder_->DrawRectangle(image, facedata);
}

// 设置欧氏距离阈值
bool DlibRecognizer::setThreshold(double threshold)
{
    // 这里可以根据需要调整阈值
    this->tolerance_ = threshold;
    LOGI("设置识别阈值为:" << threshold);
    return threshold == this->tolerance_;
}

// 获取当前使用的后端名称
std::string DlibRecognizer::getBackendName() const
{
    std::string backname = "dlib";
    return backname;
}

DlibRecognizer::~DlibRecognizer() {
}
