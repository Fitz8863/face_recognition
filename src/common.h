#pragma once

#include <iostream>
#include <string.h>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// 优化后的日志宏
#define LOGI(x) std::cout << "[INFO] [" << __FUNCTION__ << "] " << x << std::endl
#define LOGW(x) std::cout << "[WARN] [" << __FILENAME__ << ":" << __LINE__ << " " << __FUNCTION__ << "] " << x << std::endl
#define LOGE(x) std::cerr << "[ERRO] [" << __FILENAME__ << ":" << __LINE__ << " " << __FUNCTION__ << "] " << x << std::endl

// -------------------------人脸库路径-----------------------------------------
#define DATABASE_PATH "/home/fitz/projects/face/opencv_face_recognition/data/database/face.db"

// ------------------------------------------------------------------
// 人脸识别模式枚举
typedef enum
{
    OPENCV,
    DLIB,
    INSPIREFACE
} Type;

// 人脸数据结构体
typedef struct Facedata
{
    int id = -1;
    int x, y, width, height;      // 人脸框 (x, y, w, h)
    float score = 0.0f;           // 检测分数
    std::vector<float> embedding; // 128维或512维特征向量
    std::string name;             // 识别出的姓名
} Facedata;

// 
typedef struct FaceStateInfo
{
    float rgb_liveness = -1.f;
    float mask = -1.f;
    float quality = -1.f;
    struct FaceAttributeResult
    {
        int32_t race;
        int32_t gender;
        int32_t ageBracket;
    } attribute;
    struct Face3DAngle
    {
        float pitch;
        float roll;
        float yaw;
    } face3DAngle;
    struct FaceEmotionResult
    {
        int32_t emotion;
    } emotion;
    struct FaceInteractionState
    {
        float left_eye_status_confidence;
        float right_eye_status_confidence;
    } eye_state;
    struct FaceInteractionAction
    {
        int32_t normal;    ///< Normal action.
        int32_t shake;     ///< Shake action.
        int32_t jawOpen;   ///< Jaw open action.
        int32_t headRaise; ///< Head raise action.
        int32_t blink;     ///< Blink action.
    } interaction_action;
}FaceStateInfo;