#include "FaceRecognizer.h"
#include <vector>
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
int main(int argc, char const *argv[])
{
    std::string backend = argv[1];
    Type type = INSPIREFACE;
    if (backend == "dlib")
    {
        type = DLIB;
    }
    else if (backend == "opencv")
    {
        type = OPENCV;
    }
    else if (backend == "inspireface")
    {
        type = INSPIREFACE;
    }

    auto recognizer = FaceRecognizer::create(type);

    // // 遍历目录中的所有文件,批量注册人脸
    // std::string image_dir = "/home/fitz/projects/face/opencv_face_recognition/data/register_face_images";
    // std::vector<std::string> extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"};
    // for (const auto &entry : fs::directory_iterator(image_dir))
    // {
    //     if (entry.is_regular_file())
    //     {
    //         std::string ext = entry.path().extension().string();
    //         if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end())
    //         {
    //             // 打印不含扩展名的文件名
    //             std::string nameWithoutExt = entry.path().stem().string();
    //             // std::cout << nameWithoutExt << std::endl;

    //             // 使用OpenCV加载图像
    //             cv::Mat img = cv::imread(entry.path().string());
    //             if (img.empty())
    //             {
    //                 std::cerr << "Warning: Could not load image: " << nameWithoutExt << std::endl;
    //             }
    //             else
    //             {
    //                 // 注册人脸
    //                 recognizer->registerFace(img, nameWithoutExt);
    //             }
    //         }
    //     }
    // }

    std::cout << "人脸库人脸的数量: " << recognizer->getFacedatabaseCount() << " faces." << std::endl;

    // 实时摄像头识人脸
    cv::VideoCapture cap(2);
    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (cap.read(frame))
    {

        std::vector<Facedata> face = recognizer->recognizeFace(frame);

        // auto faceinfo = recognizer->Alivedetect(frame);
        // if (faceinfo)
        // {
        //     std::cout<< faceinfo->attribute.gender << std::endl;
        // }

        recognizer->drawFaceBoxes(frame, face);

        frame_count++;
        auto current_time = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count() / 1000.0;
        if (elapsed_seconds >= 1.0)
        {
            std::cout << "实时 FPS: " << frame_count / elapsed_seconds << std::endl;
            frame_count = 0;
            start_time = current_time;
        }

        cv::imshow("frame", frame);
        cv::waitKey(1);
    }
    return 0;
}
