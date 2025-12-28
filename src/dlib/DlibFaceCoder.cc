#include "DlibFaceCoder.h"

DlibFaceCoder::DlibFaceCoder(
    const std::string& shape_predictor_path,
    const std::string& face_recognition_model_path)
{
    this->loadModels(shape_predictor_path, face_recognition_model_path);
}

// 工厂方法
std::unique_ptr<DlibFaceCoder> DlibFaceCoder::create(const std::string& shape_predictor_path,
    const std::string& face_recognition_model_path)
{
    std::unique_ptr<DlibFaceCoder> dlib_face_coder = std::make_unique<DlibFaceCoder>(shape_predictor_path, face_recognition_model_path);
    return dlib_face_coder;
}

// 加载模型
bool DlibFaceCoder::loadModels(const std::string& shape_predictor_path,
    const std::string& face_recognition_model_path)
{
    // 加载人脸检测器 (HOG + SVM)
    this->detector_ = get_frontal_face_detector();

    // 加载人脸关键点检测器
    deserialize(shape_predictor_path) >> this->sp_;

    // 加载深度学习人脸识别模型
    deserialize(face_recognition_model_path) >> this->net_;

    return true;
}

// 类型转换函数,dlib人脸特征向量转为vector中
std::vector<float> DlibFaceCoder::Matrix2Vector(matrix<float, 0, 1>& encoding)
{
    std::vector<float> vec(
        encoding.begin(),
        encoding.end()
    );
    return vec;
}

// vector转dlib矩阵
matrix<float, 0, 1> DlibFaceCoder::Vector2Matrix(std::vector<float> vec)
{
    dlib::matrix<float, 0, 1> encoding = dlib::mat(vec);
    return encoding;
}

// dlib图像预处理函数
matrix<rgb_pixel> DlibFaceCoder::preprocess_image(const matrix<rgb_pixel>& img)
{
    if (img.nc() <= MAX_IMAGE_WIDTH && img.nr() <= MAX_IMAGE_HEIGHT)
    {
        this->scale_ = 1.0;
        return img;
    }

    this->scale_ = std::min(static_cast<double>(MAX_IMAGE_WIDTH) / img.nc(),
        static_cast<double>(MAX_IMAGE_HEIGHT) / img.nr());


    // 计算新的尺寸
    int new_width = static_cast<int>(img.nc() * this->scale_);
    int new_height = static_cast<int>(img.nr() * this->scale_);
    matrix<rgb_pixel> resized_img;
    resized_img.set_size(new_height, new_width);
    dlib::resize_image(img, resized_img, dlib::interpolate_bilinear());

    return resized_img;
}

// OpenCV图像预处理
cv::Mat DlibFaceCoder::preprocess_image(const cv::Mat& cv_img)
{
    cv::Mat resized_img, rgb_img;
    if (cv_img.cols <= MAX_IMAGE_WIDTH && cv_img.rows <= MAX_IMAGE_HEIGHT)
    {
        resized_img = cv_img;
        this->scale_ = 1.0;
    }
    else {
        this->scale_ = std::min(static_cast<double>(MAX_IMAGE_WIDTH) / cv_img.cols,
            static_cast<double>(MAX_IMAGE_WIDTH) / cv_img.rows);
        cv::resize(cv_img, resized_img, cv::Size(), this->scale_, this->scale_);
    }

    // 关键步骤：OpenCV 是 BGR，dlib 期待 RGB
    if (resized_img.channels() == 3)
    {
        cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);
    }
    else if (resized_img.channels() == 1)
    {
        cv::cvtColor(resized_img, rgb_img, cv::COLOR_GRAY2RGB);
    }
    else
    {
        rgb_img = resized_img; // 处理其他情况
    }

    return rgb_img;
}


// 人脸检测，返回人脸的方框列表
std::vector<dlib::full_object_detection> DlibFaceCoder::detect_faces(const cv_image<rgb_pixel>& img)
{
    std::vector<dlib::full_object_detection> shapes;

    // 使用 dlib 的 HOG + SVM 检测所有人脸
    auto faces = this->detector_(img);
    // 获取人脸特征68/5点
    for (auto& face : faces)
    {
        dlib::full_object_detection shape = this->sp_(img, face);
        shapes.push_back(shape);
    }
    return shapes;
}

// 从 OpenCV 图像获取所有人脸数据,初始化操作
std::vector<Facedata> DlibFaceCoder::get_facedatas(const cv::Mat& cv_img)
{
    cv::Mat rgb_img;

    // 预处理 OpenCV 图像
    rgb_img = preprocess_image(cv_img);

    // 将 cv::Mat 包装成 dlib 格式（这里是浅拷贝，不产生额外开销）
    cv_image<rgb_pixel> dlib_img(rgb_img);

    std::vector<Facedata> face_datas;

    auto shapes = this->detect_faces(dlib_img);

    // 2. 为每张人脸提取编码
    for (const auto& shape : shapes)
    {
        // 裁剪并标准化人脸 (150x150, 适当填充)
        matrix<rgb_pixel> face_chip;

        extract_image_chip(dlib_img, get_face_chip_details(shape, 150, 0.25), face_chip);

        // 提取128维特征向量 (人脸编码)
        matrix<float, 0, 1> encoding = this->net_(face_chip);

        // 获取人脸矩形框
        dlib::rectangle rect = shape.get_rect();

        // 构建 Facedata 结构体
        Facedata fd;
        fd.id = -1;
        fd.name = "unknown";
        fd.x = rect.left();
        fd.y = rect.top();
        fd.width = rect.right() - rect.left();
        fd.height = rect.bottom() - rect.top();
        fd.score = 0.0;
        fd.embedding = this->Matrix2Vector(encoding); // 将矩阵转换为向量
        face_datas.push_back(fd);
    }
    return face_datas;
}

// 已有人脸库与单个人脸进行比较，判断是否匹配 (默认阈值0.6),输入图片中可能有多张人脸，多个人脸全部匹配成功才返回True
double DlibFaceCoder::compareFeatures(const Facedata& face1, const Facedata& face2)
{
    // 计算欧氏距离 (与Python face_recognition库相同)
    matrix<float, 0, 1> encoding1 = this->Vector2Matrix(face1.embedding);
    matrix<float, 0, 1> encoding2 = this->Vector2Matrix(face2.embedding);
    double distance = dlib::length(encoding1 - encoding2);
    return distance;
}

// 绘制人脸框,opencv格式
void DlibFaceCoder::DrawRectangle(cv::Mat& img, std::vector<Facedata> face_datas)
{
    // std::cout << "缩放系数" << this->scale_ << std::endl;
    for (const auto& face : face_datas)
    {
        int x = face.x;
        int y = face.y;
        int w = face.width;
        int h = face.height;
        cv::Scalar color = (face.name == "unknown") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::rectangle(img, cv::Rect(x, y, w, h), color, 2);
        cv::putText(img, face.name, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    }
}

DlibFaceCoder::~DlibFaceCoder()
{
    // 析构函数，释放资源（如果有需要）
}