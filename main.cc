#include "FaceRecognizer.h"
#include <vector>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;
int main(int argc, char const *argv[])
{

    auto recognizer = FaceRecognizer::create(OPENCV);
    recognizer->registerFace("/home/fitz/projects/face/opencv_face_recognition/data/test_image/hwx.jpg", "hwx");

    // std::string image_dir = "/home/fitz/projects/face/opencv_face_recognition/data/register_face_images";
    // std::vector<std::string> extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"};

    // // 遍历目录中的所有文件
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

    std::cout << "face_count: " << recognizer->getFacedatabaseCount() << " faces." << std::endl;

    // std::string input_path = argv[1];

    // cv::Mat img = cv::imread(input_path);
    // std::vector<Facedata> face = recognizer->recognizeFace(img);
    // recognizer->drawFaceBoxes(img, face);
    // std::cout << face[0].name << std::endl;
    // cv::imwrite("output.png", img);

    

    cv::VideoCapture cap(2);
    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (cap.read(frame))
    {
        std::vector<Facedata> face = recognizer->recognizeFace(frame);
        recognizer->drawFaceBoxes(frame, face);
        cv::imshow("frame", frame);
        cv::waitKey(1);

        frame_count++;
        auto current_time = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count() / 1000.0;
        if (elapsed_seconds >= 1.0)
        {
            std::cout << "实时 FPS: " << frame_count / elapsed_seconds << std::endl;
            frame_count = 0;
            start_time = current_time;
        }
    }

    return 0;
}
