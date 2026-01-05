#include "FaceRecognizer.h"

#if defined(FACE_BACKEND_OPENCV)
#include "opencv/OpencvRecognizer.h"
#elif defined(FACE_BACKEND_DLIB)
#include "dlib/DlibRecognizer.h"
#elif defined(FACE_BACKEND_INSPIREFACE)
#include "inspireface/InspireFaceRecognizer.h"
#endif

std::unique_ptr<FaceRecognizer> FaceRecognizer::create(Type type)
{
    std::unique_ptr<FaceRecognizer> recognizer = nullptr;
    switch (type)
    {
#if defined(FACE_BACKEND_OPENCV)
    case Type::OPENCV:
        recognizer = std::make_unique<OpencvRecognizer>(
            DATABASE_PATH,
            OPENCV_DETECTOR_PATH,
            OPENCV_RECOGNIZER_PATH);
        LOGI("Using OpenCV");
        break;
#elif defined(FACE_BACKEND_DLIB)
    case Type::DLIB:
        recognizer = std::make_unique<DlibRecognizer>(
            DATABASE_PATH,
            DLIB_DETECTOR_PATH,
            DLIB_RECOGNIZER_PATH);
        LOGI("Using Dlib");
        break;
#elif defined(FACE_BACKEND_INSPIREFACE)
    case Type::INSPIREFACE:
        recognizer = std::make_unique<InspireFaceRecognizer>(
            DATABASE_PATH,
            MODEL_PATH);
        LOGI("Using InspireFace");
        break;
#endif
    default:
        break;
    }

    return recognizer;
}


