#include "FaceManager.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{

    // std::string test_image_path = argv[1];
    // if (0 == test_image_path.size())
    // {
    //     std::cout << "请提供测试图像路径作为命令行参数" << std::endl;
    //     return -1;
    // }

    // 初始化人脸识别器 (替换为您的模型路径)
    FaceCoder facecoder(
        "../models/face_detection_yunet_2023mar_int8.onnx",
        "../models/shape_predictor_5_face_landmarks.dat",
        "../models/dlib_face_recognition_resnet_model_v1.dat");

    // 加载人脸数据库
    FaceDatabase face_db("../data/database/face.db");

    // 创建人脸manager指针
    auto manager = std::make_shared<FaceManager>(face_db, facecoder);

    // manager->register_face("/home/fitz/projects/face/test/test/data/register_face_images/trump.png", "trump");

    // std::string image_dir = "/home/fitz/projects/face/test/test/data/register_face_images";
    // std::vector<std::string> extensions = { ".jpg", ".jpeg", ".JPG", ".JPEG" };

    // 遍历目录中的所有文件
    // for (const auto& entry : fs::directory_iterator(image_dir)) {
    //     if (entry.is_regular_file()) {
    //         std::string ext = entry.path().extension().string();
    //         if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
    //             // 打印不含扩展名的文件名
    //             std::string nameWithoutExt = entry.path().stem().string();
    //             // std::cout << nameWithoutExt << std::endl;

    //             // 使用OpenCV加载图像
    //             cv::Mat img = cv::imread(entry.path().string());
    //             if (img.empty()) {
    //                 std::cerr << "Warning: Could not load image: " << nameWithoutExt << std::endl;
    //             }
    //             else {
    //                 // 注册人脸
    //                 manager->register_face(img, nameWithoutExt);
    //             }
    //         }
    //     }
    // }

    // // 看看数据库人脸数量
    std::cout << "数据库人脸数量: " << manager->get_databaseface_count() << std::endl;

    cv::VideoCapture cap(0);
    

    if (!cap.isOpened())
    {
        std::cerr << "Error: Cannot open camera" << std::endl;
        return -1;
    }

    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    while (true)
    {
        cv::Mat frame;
        std::vector<FaceData> faces;
        cap >> frame; // 捕获一帧图像
        if (frame.empty())
        {
            std::cerr << "Error: Empty frame" << std::endl;
            break;
        }

        // 比较人脸
        manager->compare_face(frame, faces, 0.6);

        // 绘制人脸框
        manager->draw_rectangle(frame, faces);

        cv::imshow("Face Recognition", frame);
        if (cv::waitKey(1) == 27)
        { // 按下 'Esc' 键退出
            break;
        }

        frame_count++;
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        if (elapsed_seconds.count() >= 1.0)
        {
            std::cout << "FPS: " << frame_count / elapsed_seconds.count() << std::endl;
            frame_count = 0;
            start_time = current_time;
        }
    }

    // 读取测试图像并获取人脸数据

    // std::vector<FaceData> faces;
    // cv::Mat img = cv::imread(test_image_path);

    // // 比较人脸
    // manager->compare_face(img, faces, 0.6);

    // // 绘制人脸框
    // manager->draw_rectangle(img, faces);

    // std::cout << "识别的名字" << faces[0].name << std::endl;
    // cv::imwrite("output.jpg", img);

    return 0;
}