#include "OpencvFaceCoder.h"

OpencvFaceCoder::OpencvFaceCoder(const std::string& detectorPath,
    const std::string& recognizerPath)
{
    this->loadModels(detectorPath, recognizerPath);
}

// 工厂方法
std::unique_ptr<OpencvFaceCoder> OpencvFaceCoder::create(const std::string& detectorPath,
        const std::string& recognizerPath) {
    std::unique_ptr<OpencvFaceCoder> ptr = std::make_unique<OpencvFaceCoder>(detectorPath, recognizerPath);
    return ptr;
}

// 加载模型
bool OpencvFaceCoder::loadModels(const std::string& detectorPath,
    const std::string& recognizerPath)
{
    // 创建人脸检测器
    this->detector_ = cv::FaceDetectorYN::create(detectorPath, "", cv::Size(DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE), DETECTOR_CONFIDENCE_THRESHOLD, DETECTOR_NMS_THRESHOLD, DETECTOR_TOPK, backend_target_pairs[backendId_].first, backend_target_pairs[backendId_].second);
    if (detector_.empty())
    {
        LOGE("无法加载人脸检测模型: " << detectorPath);
        return false;
    }
    // 创建人脸识别器
    this->recognizer_ = cv::FaceRecognizerSF::create(recognizerPath, "", backend_target_pairs[backendId_].first, backend_target_pairs[backendId_].second);
    if (recognizer_.empty())
    {
        LOGE("无法加载人脸识别模型: " << recognizerPath);
        return false;
    }
    return true;
}

// 类型转换 , cv::Mat 转 std::vector<float>
std::vector<float> OpencvFaceCoder::Mat2Vector(const cv::Mat& mat)
{
    std::vector<float> vec;
    cv::Mat continuousMat = mat;
    if (!mat.isContinuous()) {
        continuousMat = mat.clone(); // 确保内存连续
    }
    vec.assign((float*)continuousMat.datastart, (float*)continuousMat.dataend);
    return vec;
}
// std::vector<float> 转 cv::Mat
cv::Mat OpencvFaceCoder::Vector2Mat(const std::vector<float>& vec)
{
    cv::Mat mat(1, static_cast<int>(vec.size()), CV_32F);
    std::memcpy(mat.data, vec.data(), vec.size() * sizeof(float));
    return mat;
}

// 图像预处理,分辨率调整
cv::Mat OpencvFaceCoder::preprocessImage(const cv::Mat& image)
{
    cv::Mat resizedImage;
    if (image.cols <= MAX_INPUT_WIDTH && image.rows <= MAX_INPUT_HEIGHT)
    {
        this->scale_ = 1.0;
        return image;
    }

    this->scale_ = std::min(static_cast<double>(MAX_INPUT_WIDTH) / image.cols,
        static_cast<double>(MAX_INPUT_HEIGHT) / image.rows);
    cv::resize(image, resizedImage, cv::Size(), this->scale_, this->scale_);
    return resizedImage;
}

// 人脸检测, 返回mat格式的人脸框信息
cv::Mat OpencvFaceCoder::detectFaces(const cv::Mat& image)
{
    cv::Mat faces;
    this->detector_->setInputSize(image.size());
    this->detector_->detect(image, faces);
    return faces;
}

// 人脸特征提取（一个图片可能有多个人脸），返回Facedata数组
std::vector<Facedata> OpencvFaceCoder::get_facedatas(const cv::Mat& image)
{
    std::vector<Facedata> facedatas;
    // 1.预处理图像
    cv::Mat preprocessedImage = this->preprocessImage(image);

    // 2.检测人脸
    cv::Mat faces = this->detectFaces(preprocessedImage);

    // 3.提取特征
    // 如果没有检测到人脸，返回空向量
    if (faces.rows == 0)
    {
        // LOGE("未检测到人脸，无法提取特征。");
        return facedatas;
    }
    // 正常提取每张人脸的特征
    else
    {
        for (int i = 0; i < faces.rows; ++i)
        {
            cv::Mat alignedFace;
            cv::Mat feature;
            recognizer_->alignCrop(preprocessedImage, faces.row(i), alignedFace);
            recognizer_->feature(alignedFace, feature);
            Facedata facedata;
            facedata.id = -1;
            // 调整坐标到原始图像尺度
            facedata.x = static_cast<int>(faces.at<float>(i, 0) / this->scale_);
            facedata.y = static_cast<int>(faces.at<float>(i, 1) / this->scale_);
            facedata.width = static_cast<int>(faces.at<float>(i, 2) / this->scale_);
            facedata.height = static_cast<int>(faces.at<float>(i, 3) / this->scale_);

            facedata.name = "unknown"; // 默认名称
            facedata.score = 0.0f; // 人脸检测的分数
            facedata.embedding = this->Mat2Vector(feature.clone()); // 深拷贝

            // 转换特征为 std::vector<float>
            facedatas.push_back(facedata);
        }
    }
    return facedatas;
}

// 两个人脸特征进行比较计算, 返回相似度分数
double OpencvFaceCoder::compareFeatures(const Facedata& face1, const Facedata& face2)
{
    // cv::Mat feature1 = face1.embedding;
    // cv::Mat feature2 = face2.embedding;
    cv::Mat feature1 = this->Vector2Mat(face1.embedding);
    cv::Mat feature2 = this->Vector2Mat(face2.embedding);
    double score = this->recognizer_->match(feature1, feature2);
    return score;
}

// 在图像上绘制人脸框
void OpencvFaceCoder::drawFaceBoxes(cv::Mat& image, std::vector<Facedata>& facedata) {
    for (const auto& face : facedata) {
        int x = face.x;
        int y = face.y;
        int w = face.width;
        int h = face.height;

        cv::Scalar color = (face.name == "unknown") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::rectangle(image, cv::Rect(x, y, w, h), color, 2);
        cv::putText(image, face.name, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    }
}


