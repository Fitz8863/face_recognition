#pragma once
#include <opencv2/opencv.hpp>
#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>
#include <usearch/index_dense.hpp>

#include <iostream>
#include <string.h>
#include <memory>
#include <string>
#include <vector>

using namespace unum::usearch;

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
    DLIB
} Type;

// 人脸数据结构体
typedef struct Facedata
{
    int id;
    int x, y, width, height;      // 人脸框 (x, y, w, h)
    float score = 0.0f;           // 检测分数
    std::string name;             // 识别出的姓名
    std::vector<float> embedding; // 128维（或512维）特征向量
} Facedata;
