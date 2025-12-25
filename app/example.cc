

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <chrono>

// 1. 定义后端和目标设备的映射 (参考 recog.cpp)
const std::vector<std::pair<int, int>> backend_target_pairs = {
    {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},       // 0: CPU
    {cv::dnn::DNN_BACKEND_CUDA,   cv::dnn::DNN_TARGET_CUDA},      // 1: CUDA
    {cv::dnn::DNN_BACKEND_CUDA,   cv::dnn::DNN_TARGET_CUDA_FP16}, // 2: CUDA FP16
    {cv::dnn::DNN_BACKEND_TIMVX,  cv::dnn::DNN_TARGET_NPU},       // 3: TIM-VX (NPU)
    {cv::dnn::DNN_BACKEND_CANN,   cv::dnn::DNN_TARGET_NPU}        // 4: CANN (NPU)
};

class FaceDatabase {
public:
    void registerFace(const std::string &name, const cv::Mat &feature) {
        db[name] = feature.clone();
        std::cout << "注册成功: " << name << std::endl;
    }

    std::string search(cv::Ptr<cv::FaceRecognizerSF> &recognizer, const cv::Mat &queryFeature, double threshold = 0.363) {
        std::string bestName = "Unknown";
        double maxScore = -1.0;
        for (auto const &[name, faceFeature] : db) {
            double score = recognizer->match(queryFeature, faceFeature, cv::FaceRecognizerSF::DisType::FR_COSINE);
            std::cout<<"比对 " << name << " 得分: " << score << std::endl;
            if (score > maxScore) {
                maxScore = score;
                if (score >= threshold) {
                    bestName = name;
                }
            }
        }
        return bestName;
    }
    size_t size() const { return db.size(); }
private:
    std::map<std::string, cv::Mat> db;
};

int main(int argc, char** argv) {

    int device_idx = 0;
    std::string detector_path = "/home/fitz/projects/face/opencv_face_recognition/src/opencv_impl/models/face_detection_yunet_2023mar_int8.onnx";
    std::string recognizer_path = "/home/fitz/projects/face/opencv_face_recognition/src/opencv_impl/models/face_recognition_sface_2021dec_int8.onnx";

    // 获取对应的 backend 和 target ID
    const int backend_id = 0;//backend_target_pairs.at(device_idx).first;
    const int target_id =0; //backend_target_pairs.at(device_idx).second;

    std::cout << "正在启动，使用后端 ID: " << backend_id << ", 目标设备 ID: " << target_id << std::endl;

    // 3. 在创建实例时传入 backend_id 和 target_id
    // YuNet 创建
    auto detector = cv::FaceDetectorYN::create(detector_path, "", cv::Size(320, 320), 0.6f, 0.3f, 5000, backend_id, target_id);
    // SFace 创建
    auto recognizer = cv::FaceRecognizerSF::create(recognizer_path, "", backend_id, target_id);

    FaceDatabase myDB;

    // --- 【注册逻辑】 ---
    cv::Mat regImg = cv::imread("/home/fitz/projects/face/opencv_face_recognition/data/register_face_images/hwj1.jpg");
    if (!regImg.empty()) {
        float scale = 640.0 / regImg.rows;
        cv::resize(regImg, regImg, cv::Size(), scale, scale);
        detector->setInputSize(regImg.size());
        cv::Mat faces;
        detector->detect(regImg, faces);
        if (faces.rows > 0) {
            cv::Mat alignedFace, feature;
            recognizer->alignCrop(regImg, faces.row(0), alignedFace);
            recognizer->feature(alignedFace, feature);
            myDB.registerFace("hwj", feature);
        }
    }

    cv::Mat regImg2 = cv::imread("/home/fitz/projects/face/opencv_face_recognition/data/register_face_images/034.jpg");
    if (!regImg2.empty()) {
        float scale = 640.0 / regImg2.rows;
        cv::resize(regImg2, regImg2, cv::Size(), scale, scale);
        detector->setInputSize(regImg2.size());
        cv::Mat faces;
        detector->detect(regImg2, faces);
        if (faces.rows > 0) {
            cv::Mat alignedFace, feature;
            recognizer->alignCrop(regImg2, faces.row(0), alignedFace);
            recognizer->feature(alignedFace, feature);
            myDB.registerFace("034", feature);
        }
    }

    // --- 【实时识别逻辑】 ---
    cv::VideoCapture cap(0);
    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    // while (cv::waitKey(1) < 0) {
        // cap >> frame;
    frame = cv::imread("/home/fitz/projects/face/opencv_face_recognition/data/test_image/hwj2_2.png");
        // if (frame.empty()) break;

        detector->setInputSize(frame.size());
        cv::Mat faces;
        detector->detect(frame, faces);

        for (int i = 0; i < faces.rows; i++) {
            cv::Mat alignedFace, queryFeature;
            recognizer->alignCrop(frame, faces.row(i), alignedFace);
            recognizer->feature(alignedFace, queryFeature);

            std::string identity = myDB.search(recognizer, queryFeature);

            float x = faces.at<float>(i, 0);
            float y = faces.at<float>(i, 1);
            float w = faces.at<float>(i, 2);
            float h = faces.at<float>(i, 3);

            cv::Scalar color = (identity == "Unknown") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(frame, cv::Rect(x, y, w, h), color, 2);
            cv::putText(frame, identity, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        }

        // FPS 计算
        frame_count++;
        auto current_time = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count() / 1000.0;
        if (elapsed_seconds >= 1.0) {
            std::cout << "实时 FPS: " << frame_count / elapsed_seconds << std::endl;
            frame_count = 0;
            start_time = current_time;
        }

    //     cv::imshow("OpenCV YuNet + SFace", frame);
    // }
    return 0;
}
