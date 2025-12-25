#include "common.h"

// -------------------------opencv检测器和识别器路径--------------------------------

#define OPENCV_DETECTOR_PATH "/home/fitz/projects/face/opencv_face_recognition/models/opencv/face_detection_yunet_2023mar.onnx"
#define OPENCV_RECOGNIZER_PATH "/home/fitz/projects/face/opencv_face_recognition/models/opencv/face_recognition_sface_2021dec.onnx"

// --------------------------------------------------------------------------

// 选择使用的后端和目标设备
#define BACKEND_ID 0 // 0: CPU, 1: CUDA, 2: CUDA FP16, 3: TIM-VX (NPU)
#define TARGET_ID 0 // 0: CPU, 1: CUDA, 2: CUDA FP16, 3: NPU

// 人脸检测模型初始化参数
#define DETECTOR_INPUT_SIZE 640
#define DETECTOR_CONFIDENCE_THRESHOLD 0.7 // 人脸检测阈值
#define DETECTOR_NMS_THRESHOLD 0.3 // 非极大抑制阈值
#define DETECTOR_TOPK 5000

// 人脸识别模型初始化参数
#define RECOGNIZER_INPUT_SIZE 112
#define RECOGNIZER_CONFIDENCE_THRESHOLD 0.45 // 初始化阈值 0.364 对应欧氏距离 1.128

// 预处理输入图片大小的限制
#define MAX_INPUT_WIDTH 720
#define MAX_INPUT_HEIGHT 720

const std::vector<std::pair<int, int>> backend_target_pairs = {
    {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},       // 0: CPU
    {cv::dnn::DNN_BACKEND_CUDA,   cv::dnn::DNN_TARGET_CUDA},      // 1: CUDA
    {cv::dnn::DNN_BACKEND_CUDA,   cv::dnn::DNN_TARGET_CUDA_FP16}, // 2: CUDA FP16
    {cv::dnn::DNN_BACKEND_TIMVX,  cv::dnn::DNN_TARGET_NPU},       // 3: TIM-VX (NPU)
    // {cv::dnn::DNN_BACKEND_CANN,   cv::dnn::DNN_TARGET_NPU}        // 4: CANN (NPU)
};
