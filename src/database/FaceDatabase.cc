#include "FaceDatabase.h" 
#include "OpencvFaceDatabase.h"
#include "DlibFaceDatabase.h"

// 工厂模式
std::unique_ptr<FaceDatabase> FaceDatabase::create(const std::string& db_path,Type type)
{
    std::unique_ptr<FaceDatabase> facedatabase;
    switch (type)
    {
    case Type::OPENCV:
        facedatabase = std::make_unique<OpencvFaceDatabase>(db_path);
        // LOGI("Using OpenCV FaceDatabase");
        break;
    case Type::DLIB:
        facedatabase = std::make_unique<DlibFaceDatabase>(db_path);
        // LOGI("Using Dlib FaceDatabase");
        break;
    default:
        return nullptr;
    }
    return facedatabase;
}