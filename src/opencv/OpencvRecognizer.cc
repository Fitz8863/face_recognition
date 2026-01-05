#include "OpencvRecognizer.h"

// 构造函数
OpencvRecognizer::OpencvRecognizer(const std::string &dbPath,
                                   const std::string &detectorPath,
                                   const std::string &recognizerPath)
{
    this->facedatabase_ = FaceDatabase::create(dbPath, OPENCV);
    this->facecoder_ = OpencvFaceCoder::create(detectorPath, recognizerPath);

    // 加载人脸数据库中的所有人脸数据到内存
    std::vector<Facedata> facedatas = facedatabase_->load_all_faces();

    // 1. 定义度量
    metric_punned_t metric(128, metric_kind_t::cos_k, scalar_kind_t::f32_k);

    // 2. 定义配置 (可选，但建议显式指定)
    index_dense_config_t config;
    // config.connectivity = 16; // 默认 M=16，增加此值可提高精度但增加内存

    // 3. 初始化索引
    this->index_ = index_dense_t::make(metric, config);

    // 4. 预留空间（提升性能）
    this->index_.reserve(facedatas.size());

    // 5. 循环添加数据
    for (const Facedata &face : facedatas)
    {
        // 1. 加入向量索引
        // 注意：这里的 face.id 必须是正整数 (uint64_t)
        this->index_.add(face.id, face.embedding.data());

        //  额外存一个 ID 到数据的映射，方便搜索后直接取出结构体
        this->facedata_map_[face.id] = face;
    }
}

// 在人脸库中注册新的人脸（传入图片路径）
bool OpencvRecognizer::registerFace(const std::string path, const std::string &name)
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
    for (const auto &face : list)
    {
        if (face.name != "unknown")
        {
            LOGE("已存在此人脸，请勿重复注册，名字:" << face.name);
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
    Facedata &newFace = newFaces[0];
    newFace.name = registered_name;

    // 插入到数据库
    uint64_t id = facedatabase_->insert(newFace, path); // img_path 可选，这里传空字符串
    if (-1 != id)
    {
        // 同时添加到内存中的人脸数据
        this->facedata_map_[id] = newFace;
        this->index_.add(id, newFace.embedding.data());
    }
    else
    {
        LOGW("insert face failed");
        return false;
    }
    return true;
}

// 在人脸库中注册新的人脸（传入opencv 图片）
bool OpencvRecognizer::registerFace(const cv::Mat &image, const std::string &name)
{
    std::string registered_name = name;

    // 检测人脸库里面是否有这个人脸，这里做特征向量匹配
    std::vector<Facedata> list = this->recognizeFace(image);
    for (const auto &face : list)
    {
        if (face.name != "unknown")
        {
            LOGE("已存在此人脸，请勿重复注册，名字:" << face.name);
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
    Facedata &newFace = newFaces[0];
    newFace.name = registered_name;

    // 插入到数据库
    uint64_t id = this->facedatabase_->insert(newFace, ""); // img_path 可选，这里传空字符串
    if (-1 != id)
    {
        // 同时添加到内存中的人脸数据
        this->facedata_map_[id] = newFace;
        this->index_.add(id, newFace.embedding.data());
    }
    else
    {
        LOGE("插入失败");
        return false;
    }
    return true;
}

// 在人脸库匹配图片的人脸特征，返回对应的人脸结构列表
std::vector<Facedata> OpencvRecognizer::recognizeFace(const cv::Mat &faceImage)
{
    std::vector<Facedata> queryFaces = this->facecoder_->get_facedatas(faceImage);

    if (queryFaces.empty())
    {
        // LOGE("未检测到人脸，识别失败");
        return queryFaces;
    }

    // 当前人脸查找方式（使用向量索引查找）
    for (auto &queryFace : queryFaces)
    {
        if (this->index_.size() <= 0)
        {
            return queryFaces;
        }
        auto results = this->index_.search(queryFace.embedding.data(), 3);

        for (size_t i = 0; i < results.size(); ++i)
        {
            uint64_t found_id = results[i].member.key; // 之前 add 进去的 ID
            float distance = results[i].distance;      // 余弦距离  注意：余弦距离(Distance) = 1 - 余弦相似度(Similarity)，距离越小（接近0），代表越相似

            Facedata match = this->facedata_map_[found_id];                          //  获取之前 add 进去的 Facedata
            double similarity = this->facecoder_->compareFeatures(queryFace, match); //  计算余弦相似度
            if (similarity >= this->threshold_)
            {
                queryFace.id = match.id;
                queryFace.name = match.name;
                queryFace.score = distance;
            }
        }
    }

    // 原本的查找方式(一个个遍历计算余弦相似度)

    // for (auto &queryFace : queryFaces)
    // {
    //     double bestScore = 0.0;
    //     Facedata bestMatch;
    //     for (const auto &[_, dbFace] : this->facedata_map_)
    //     {
    //         double score = facecoder_->compareFeatures(queryFace, dbFace);
    //         if (score > bestScore)
    //         {
    //             bestScore = score;
    //             bestMatch = dbFace;
    //         }
    //         // 如果相似度超过阈值，认为是同一人
    //         if (bestScore >= this->threshold_)
    //         {
    //             // 更新查询人脸的信息为匹配到的数据库人脸
    //             queryFace.id = bestMatch.id;
    //             queryFace.name = bestMatch.name;
    //             queryFace.score = bestScore;
    //             break;
    //         }
    //     }
    // }

    return queryFaces;
}

// 通过name查找人脸
std::vector<Facedata> OpencvRecognizer::findByNname(const std::string &name)
{
    return this->facedatabase_->find_by_name(name);
}

// 删除人脸操作
bool OpencvRecognizer::deleteFaceByName(const std::string &name)
{
    // 从数据库删除
    uint64_t id = facedatabase_->delete_by_name(name);
    if (-1 != id)
    {
        // 同时从内存中删除
        this->facedata_map_.erase(id);
        this->index_.remove(id);
    }
    else
    {
        LOGE("删除人脸失败: {}" << name);
        return false;
    }
    return true;
}

// 查看人脸数据库中人脸数量
int OpencvRecognizer::getFacedatabaseCount()
{
    return facedatabase_->get_face_count();
}

// 绘制人脸框
void OpencvRecognizer::drawFaceBoxes(cv::Mat &image, std::vector<Facedata> &facedata)
{
    facecoder_->drawFaceBoxes(image, facedata);
}

// 设置阈值
bool OpencvRecognizer::setThreshold(double threshold)
{
    // 这里可以根据需要调整阈值
    this->threshold_ = threshold;
    LOGI("设置新的识别阈值为:" << threshold);
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
