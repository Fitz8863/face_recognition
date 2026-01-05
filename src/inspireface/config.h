#include "common.h"
#include <opencv2/opencv.hpp>
#include "usearch/index.hpp"
#include "usearch/index_plugins.hpp"
#include "usearch/index_dense.hpp"
#include "FaceRecognizer.h"

using namespace unum::usearch;
// -------------------------opencv检测器和识别器路径--------------------------------

#define MODEL_PATH "/home/fitz/projects/face/opencv_face_recognition/models/InspireFace/Pikachu"

// --------------------------------------------------------------------------

// 人脸识别模型初始化参数
#define INSPIREFACE_DETECTOR_CONFIDENCE_THRESHOLD 0.7 // 人脸检测阈值
#define INSPIREFACE_CONFIDENCE_THRESHOLD 0.45         // 初始化阈值 0.364 对应欧氏距离 1.128

#define MAX_DETECT_FACE 100 // 最大检测人脸数
#define DETECT_LEVEL_PX 320 // 检测图片最大分辨率  160, 320, 640

// 检测参数
#define ENABLE_MASK_DETECT true // 口罩检测
#define ENABLE_RGB_LIVENESS_DETECT true // rgb活体检测
#define ENABLE_INTERACTION_LIVENESS_DETECT true // 交互活体检测
#define ENABLE_ATTRIBUTE_DETECT true // 人脸属性检测
#define ENABLE_EMOTION_DETECT true // 人脸表情检测
#define ENABLE_QUALITY_DETECT true // 人脸质量检测
#define ENABLE_INTERACTION_LIVENESS true // 人眼状态&面部动作检测


#define ENABLE_POSE_DETECT true // 人脸姿态检测
#define ENABLE_IR_LIVENESS_DETECT true // ir活体检测
#define ENABLE_RECOGNITION true // 人脸识别
