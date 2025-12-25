#include "OpencvRecognizer.h"

// 构造函数
OpencvRecognizer::OpencvRecognizer(const std::string& dbPath,
    const std::string& detectorPath,
    const std::string& recognizerPath)
{
    this->facedatabase_ = FaceDatabase::create(dbPath, OPENCV);
    this->facecoder_ = OpencvFaceCoder::create(detectorPath, recognizerPath);

    // 加载人脸数据库中的所有人脸数据到内存
    std::vector<Facedata> facedatas = facedatabase_->load_all_faces();
    for (const auto& face : facedatas)
    {
        this->facedata_map_[face.name] = face;
    }
}

// 在人脸库中注册新的人脸
bool OpencvRecognizer::registerFace(const std::string path, const std::string& name)
{
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

    // 检查是否已有相同名称的人脸，若有则添加编号后缀
    auto existing_faces = this->facedatabase_->find_by_name(name);
    int count = existing_faces.size();
    if (count > 0)
    {
        registered_name += std::to_string(count);
        LOGW("名称:" << name << "已存在，注册为新名称: " << registered_name);
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

    // 只注册第一张检测到的人脸
    Facedata& newFace = newFaces[0];
    newFace.name = registered_name;

    // 插入到数据库
    bool success = facedatabase_->insert(newFace, path); // img_path 可选，这里传空字符串
    if (success)
    {
        // 同时添加到内存中的人脸数据
        // facedatas_.push_back(newFace);
        this->facedata_map_[registered_name] = newFace;
    }
    return success;
}
// 在人脸库中注册新的人脸
bool OpencvRecognizer::registerFace(const cv::Mat& image, const std::string& name)
{
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

    // 检查是否已有相同名称的人脸，若有则添加编号后缀
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
        // facedatas_.push_back(newFace);
        this->facedata_map_[registered_name] = newFace;
    }
    return success;
}

// 在人脸库匹配图片的人脸特征，返回对应的人脸结构列表
std::vector<Facedata> OpencvRecognizer::recognizeFace(const cv::Mat& faceImage)
{
    std::vector<Facedata> queryFaces = this->facecoder_->get_facedatas(faceImage);

    if (queryFaces.empty())
    {
        // LOGE("未检测到人脸，识别失败");
        return queryFaces;
    }

    for (auto& queryFace : queryFaces)
    {
        double bestScore = 0.0;
        Facedata bestMatch;
        for (const auto& [_, dbFace] : this->facedata_map_)
        {
            double score = facecoder_->compareFeatures(queryFace, dbFace);
            if (score > bestScore)
            {
                bestScore = score;
                bestMatch = dbFace;
            }
            // 如果相似度超过阈值，认为是同一人
            if (bestScore >= this->threshold_)
            {
                // 更新查询人脸的信息为匹配到的数据库人脸
                queryFace.id = bestMatch.id;
                queryFace.name = bestMatch.name;
                queryFace.score = bestScore;
                break;
            }
            else
            {
                // LOGI("识别失败，最高相似度: " << bestScore);
            }
        }
    }
    return queryFaces;
}

// 通过name查找人脸  
std::vector<Facedata> OpencvRecognizer::findByNname(const std::string& name) {
    return this->facedatabase_->find_by_name(name);
}

// 删除人脸操作
bool OpencvRecognizer::deleteFaceByName(const std::string& name)
{
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

// 查看人脸数据库中人脸数量
int OpencvRecognizer::getFacedatabaseCount()
{
    return facedatabase_->get_face_count();
}

// 绘制人脸框
void OpencvRecognizer::drawFaceBoxes(cv::Mat& image, std::vector<Facedata>& facedata)
{
    facecoder_->drawFaceBoxes(image, facedata);
}

// 设置阈值
bool OpencvRecognizer::setThreshold(double threshold)
{
    // 这里可以根据需要调整阈值
    this->threshold_ = threshold;
    LOGI("设置识别阈值为:" << threshold);
    return threshold == this->threshold_;
}

// 获取当前使用的后端名称
std::string OpencvRecognizer::getBackendName() const
{
    std::string backname = "opencv";
    return backname;
}

// 析构函数
OpencvRecognizer::~OpencvRecognizer()
{
}
