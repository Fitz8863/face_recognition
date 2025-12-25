#pragma once

#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/geometry/vector.h>
#include <dlib/opencv.h>
#include <cmath>
#include <iomanip>
#include <algorithm>

#include "common.h"

#define MAX_IMAGE_WIDTH 640.0  // 定义图像的最大宽度
#define MAX_IMAGE_HEIGHT 640.0 // 定义图像的最大高度
#define TOLERANCE 0.50          // 欧氏距离人脸匹配容差阈值

// ----------------------------------dlib模型路径----------------------------------
#define DLIB_DETECTOR_PATH "/home/fitz/projects/face/opencv_face_recognition/models/dlib/shape_predictor_5_face_landmarks.dat"
#define DLIB_RECOGNIZER_PATH "/home/fitz/projects/face/opencv_face_recognition/models/dlib/dlib_face_recognition_resnet_model_v1.dat"
// --------------------------------------------------------------------------


// 明确使用命名空间
using namespace dlib;

// 定义与Python face_recognition库相同的ResNet架构
template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
                                                  alevel0<
                                                      alevel1<
                                                          alevel2<
                                                              alevel3<
                                                                  alevel4<
                                                                      max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

// End of file: include/common.h