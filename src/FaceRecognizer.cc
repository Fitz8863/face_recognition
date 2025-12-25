#include "FaceRecognizer.h"
#include "opencv/OpencvRecognizer.h"
#include "dlib/DlibRecognizer.h"
std::unique_ptr<FaceRecognizer> FaceRecognizer::create(Type type)
{
    std::unique_ptr<FaceRecognizer> recognizer;
    switch (type)
    {
    case OPENCV:
        recognizer = std::make_unique<OpencvRecognizer>(
            DATABASE_PATH,
            OPENCV_DETECTOR_PATH,
            OPENCV_RECOGNIZER_PATH);
            LOGI("Using OpenCV");
        break;

    case DLIB:
        recognizer = std::make_unique<DlibRecognizer>(
            DATABASE_PATH,
            DLIB_DETECTOR_PATH,
            DLIB_RECOGNIZER_PATH);
            LOGI("Using Dlib");
        break;
    default:
        break;
    }

    // 未来可以添加: "dlib", "tensorflow" 等
    // throw std::invalid_argument("Unsupported recognizer type");
    return recognizer;
}
