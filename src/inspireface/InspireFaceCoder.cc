#include "InspireFaceCoder.h"

InspireFaceCoder::InspireFaceCoder(const std::string &model_path)
{
    // Global init(only once)
    INSPIREFACE_CONTEXT->Reload(model_path);

    this->param_.enable_recognition = ENABLE_RECOGNITION;
    this->param_.enable_interaction_liveness = ENABLE_INTERACTION_LIVENESS;
    this->param_.enable_liveness = ENABLE_RGB_LIVENESS_DETECT;
    this->param_.enable_mask_detect = ENABLE_MASK_DETECT;
    this->param_.enable_face_attribute = ENABLE_ATTRIBUTE_DETECT;
    this->param_.enable_face_quality = ENABLE_QUALITY_DETECT;
    // this->param_.enable_face_pose = ENABLE_POSE_DETECT;
    this->param_.enable_face_emotion = ENABLE_EMOTION_DETECT;

    // Create a session
    auto max_detect_face = MAX_DETECT_FACE;
    auto detect_level_px = DETECT_LEVEL_PX;

    this->session_ = std::unique_ptr<inspire::Session>(
        inspire::Session::CreatePtr(
            inspire::DetectModuleMode::DETECT_MODE_ALWAYS_DETECT,
            max_detect_face,
            this->param_,
            detect_level_px));
    if (nullptr == this->session_)
    {
        LOGE("InspireFaceCoder::InspireFaceCoder() failed");
    }
}

// 工厂函数
std::unique_ptr<InspireFaceCoder> InspireFaceCoder::create(const std::string &model_path)
{
    std::unique_ptr<InspireFaceCoder> ptr = std::make_unique<InspireFaceCoder>(model_path);
    return ptr;
}

// 人脸检测
std::vector<inspire::FaceTrackWrap> InspireFaceCoder::detectFaces(const cv::Mat &image)
{
    std::lock_guard<std::mutex> lock(this->faceMutex_);
    // Create a FrameProcess for processing image formats and rotating data
    inspirecv::FrameProcess process =
        inspirecv::FrameProcess::Create(image.data, image.rows, image.cols, inspirecv::BGR, inspirecv::ROTATION_0);

    std::vector<inspire::FaceTrackWrap> results;
    int32_t ret;

    // 检测人脸，结构存储到results中
    ret = this->session_->FaceDetectAndTrack(process, results);

    return results;
}

// 人脸特征提取, 一个图片可能有多个人脸,返回Facedata数组
std::vector<Facedata> InspireFaceCoder::get_facedatas(const cv::Mat &image)
{
    std::vector<Facedata> facedatas;

    // 检测人脸
    std::vector<inspire::FaceTrackWrap> results = this->detectFaces(image);

    inspirecv::FrameProcess process =
        inspirecv::FrameProcess::Create(image.data, image.rows, image.cols, inspirecv::BGR, inspirecv::ROTATION_0);

    for (auto &result : results)
    {
        // Get face embedding
        inspire::FaceEmbedding feature;
        this->session_->FaceFeatureExtract(process, result, feature, true);

        Facedata facedata;
        facedata.id = -1;
        // 调整坐标到原始图像尺度
        facedata.x = static_cast<int>(result.rect.x / this->scale_);
        facedata.y = static_cast<int>(result.rect.y / this->scale_);
        facedata.width = static_cast<int>(result.rect.width / this->scale_);
        facedata.height = static_cast<int>(result.rect.height / this->scale_);

        facedata.name = "unknown"; // 默认名称
        facedata.score = 0.0f;     // 人脸检测的分数

        facedata.embedding = feature.embedding; // 深拷贝

        facedatas.push_back(facedata);
    }
    return facedatas;
}

// 两个人脸对比，返回余弦相似度
float InspireFaceCoder::compareFeatures(const Facedata &face1, const Facedata &face2)
{
    // Get face embedding
    float similarity = -1.0f;
    INSPIREFACE_FEATURE_HUB->CosineSimilarity(face1.embedding, face2.embedding, similarity);
    return similarity;
}

// 在图像上绘制人脸框
void InspireFaceCoder::drawFaceBoxes(cv::Mat &image, std::vector<Facedata> &facedata)
{
    for (const auto &face : facedata)
    {
        int x = face.x;
        int y = face.y;
        int w = face.width;
        int h = face.height;

        cv::Scalar color = (face.name == "unknown") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::rectangle(image, cv::Rect(x, y, w, h), color, 2);
        cv::putText(image, face.name, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    }
}

/*
执行面部流水线
要执行 Pipeline 功能，您需要先执行人脸检测或跟踪以获取 FaceTrackWrap ，选择要执行 Pipeline 的人脸作为输入参数，并为要调用的函数配置相应的选项。

所有功能都只需要一次管道接口调用，这简化了频繁调用的场景。
*/

// 人脸面部流水线初始化
void InspireFaceCoder::pipeline_init(const std::vector<inspire::FaceTrackWrap> &results, inspirecv::FrameProcess &process)
{
    int ret = this->session_->MultipleFacePipelineProcess(process, this->param_, results);
    INSPIREFACE_CHECK_MSG(ret == 0, "MultipleFacePipelineProcess failed");
}

// 人脸状态检测
std::vector<FaceStateInfo> InspireFaceCoder::StateDetect(const cv::Mat &image)
{
    std::vector<FaceStateInfo> faceinfo;

    // 检测人脸
    std::vector<inspire::FaceTrackWrap> results = this->detectFaces(image);

    inspirecv::FrameProcess process =
        inspirecv::FrameProcess::Create(image.data, image.rows, image.cols, inspirecv::BGR, inspirecv::ROTATION_0);

    if (results.size() > 0)
    {
        faceinfo.resize(results.size());
        int ret = this->session_->MultipleFacePipelineProcess(process, this->param_, results);
        INSPIREFACE_CHECK_MSG(ret == 0, "MultipleFacePipelineProcess failed");
    }
    else
    {
        faceinfo.clear();
        return faceinfo;
    }

    // 检测逻辑
    // 01--检测口罩
    if (ENABLE_MASK_DETECT == true)
    {
        std::vector<float> confidence = this->session_->GetFaceMaskConfidence();
        for (int i = 0; i < confidence.size(); i++)
        {
            faceinfo[i].mask = confidence[i];
        }
    }
    // 02--检测质量
    if (ENABLE_QUALITY_DETECT == true)
    {
        std::vector<float> confidence = this->session_->GetFaceQualityConfidence();
        for (int i = 0; i < confidence.size(); i++)
        {
            faceinfo[i].quality = confidence[i];
        }
    }
    // 03--检测rgb活体
    if (ENABLE_RGB_LIVENESS_DETECT == true)
    {
        std::vector<float> confidence = this->session_->GetRGBLivenessConfidence();
        for (int i = 0; i < confidence.size(); i++)
        {
            faceinfo[i].rgb_liveness = confidence[i];
        }
    }
    // 04--检测人脸属性
    if (ENABLE_ATTRIBUTE_DETECT == true)
    {
        std::vector<inspire::FaceAttributeResult> attribute = this->session_->GetFaceAttributeResult();
        for (int i = 0; i < attribute.size(); i++)
        {
            faceinfo[i].attribute.ageBracket = attribute[i].ageBracket;
            faceinfo[i].attribute.gender = attribute[i].gender;
            faceinfo[i].attribute.race = attribute[i].race;
        }
    }
    // 05--检测表情
    if (ENABLE_EMOTION_DETECT == true)
    {
        std::vector<inspire::FaceEmotionResult> emotion = this->session_->GetFaceEmotionResult();
        for (int i = 0; i < emotion.size(); i++)
        {
            faceinfo[i].emotion.emotion = emotion[i].emotion;
        }
    }
    // 06-07--检测交互动作(人眼&头部)
    if (ENABLE_INTERACTION_LIVENESS == true)
    {
        std::vector<inspire::FaceInteractionState> eye_state = this->session_->GetFaceInteractionState();
        std::vector<inspire::FaceInteractionAction> interaction_action = this->session_->GetFaceInteractionAction();
        for (int i = 0; i < eye_state.size(); i++)
        {
            // 眼睛状态
            faceinfo[i].eye_state.left_eye_status_confidence = eye_state[i].left_eye_status_confidence;
            faceinfo[i].eye_state.right_eye_status_confidence = eye_state[i].right_eye_status_confidence;
            
            // 交互动作
            faceinfo[i].interaction_action.blink = interaction_action[i].blink;
            faceinfo[i].interaction_action.headRaise = interaction_action[i].headRaise;
            faceinfo[i].interaction_action.jawOpen = interaction_action[i].jawOpen;
            faceinfo[i].interaction_action.normal = interaction_action[i].normal;
            faceinfo[i].interaction_action.shake = interaction_action[i].shake;
        }
    }

    return faceinfo;
}

// 人脸RGB检测
/*
RGB活体检测
RGB活体检测功能基于 RGB 图像，用于检测 RGB 图像中的人脸是否为真人。
*/
// std::vector<float> InspireFaceCoder::rgbLivenessDetect(const cv::Mat& image)
// {
//     std::vector<inspire::FaceTrackWrap> results = this->detectFaces(image);
//     inspirecv::FrameProcess process =
//         inspirecv::FrameProcess::Create(image.data, image.rows, image.cols, inspirecv::BGR, inspirecv::ROTATION_0);

//     this->pipeline_init(results, process);

//     std::vector<float> confidence = this->session_->GetRGBLivenessConfidence();
//     return confidence;
// }

// // 口罩检测
// std::vector<float> InspireFaceCoder::MaskDetect(const cv::Mat& image)
// {
//     std::vector<inspire::FaceTrackWrap> results = this->detectFaces(image);
//     inspirecv::FrameProcess process =
//         inspirecv::FrameProcess::Create(image.data, image.rows, image.cols, inspirecv::BGR, inspirecv::ROTATION_0);

//     this->pipeline_init(results, process);

//     std::vector<float> confidence = this->session_->GetFaceMaskConfidence();
//     return confidence;
// }

// // 人脸质量检测
// std::vector<float> InspireFaceCoder::QualityDetect(const cv::Mat& image)
// {
//     std::vector<inspire::FaceTrackWrap> results = this->detectFaces(image);
//     inspirecv::FrameProcess process =
//         inspirecv::FrameProcess::Create(image.data, image.rows, image.cols, inspirecv::BGR, inspirecv::ROTATION_0);

//     this->pipeline_init(results, process);

//     std::vector<float> confidence = this->session_->GetFaceQualityConfidence();
//     return confidence;
// }

// // 人脸属性检测
// std::vector<inspire::FaceAttributeResult> InspireFaceCoder::AttributeDetect(const cv::Mat& image)
// {
//     std::vector<inspire::FaceTrackWrap> results = this->detectFaces(image);
//     inspirecv::FrameProcess process =
//         inspirecv::FrameProcess::Create(image.data, image.rows, image.cols, inspirecv::BGR, inspirecv::ROTATION_0);

//     this->pipeline_init(results, process);

//     std::vector<inspire::FaceAttributeResult> attribute = this->session_->GetFaceAttributeResult();
//     return attribute;
// }

// // 面部表情检测
// std::vector<inspire::FaceEmotionResult> InspireFaceCoder::EmotionDetect(const cv::Mat& image)
// {
//     std::vector<inspire::FaceTrackWrap> results = this->detectFaces(image);
//     inspirecv::FrameProcess process =
//         inspirecv::FrameProcess::Create(image.data, image.rows, image.cols, inspirecv::BGR, inspirecv::ROTATION_0);

//     this->pipeline_init(results, process);

//     std::vector<inspire::FaceEmotionResult> emotion = this->session_->GetFaceEmotionResult();
//     return emotion;
// }

// // 眼睛状态检测
// std::vector<inspire::FaceInteractionState> InspireFaceCoder::EyeStateDetect(const cv::Mat& image)
// {
//     std::vector<inspire::FaceTrackWrap> results = this->detectFaces(image);
//     inspirecv::FrameProcess process =
//         inspirecv::FrameProcess::Create(image.data, image.rows, image.cols, inspirecv::BGR, inspirecv::ROTATION_0);

//     this->pipeline_init(results, process);

//     std::vector<inspire::FaceInteractionState> states = this->session_->GetFaceInteractionState();
//     return states;
// }

// // 面部交互动作检测
// std::vector<inspire::FaceInteractionAction> InspireFaceCoder::InteractionActionDetect(const cv::Mat& image)
// {
//     std::vector<inspire::FaceTrackWrap> results = this->detectFaces(image);
//     inspirecv::FrameProcess process =
//         inspirecv::FrameProcess::Create(image.data, image.rows, image.cols, inspirecv::BGR, inspirecv::ROTATION_0);

//     this->pipeline_init(results, process);

//     std::vector<inspire::FaceInteractionAction> action = this->session_->GetFaceInteractionAction();
//     return action;
// }

// 析构函数
InspireFaceCoder::~InspireFaceCoder()
{
}