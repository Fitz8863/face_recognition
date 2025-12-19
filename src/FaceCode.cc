#include "FaceCode.h"
#include <dlib/opencv.h>

FaceCoder::FaceCoder(const std::string &yunet_model_path,
                     const std::string &shape_predictor_path,
                     const std::string &face_recognition_model_path)
{
#ifdef USE_YUNET_DETECTOR
    // 实例化 YuNet，参数分别是：模型路径、配置、输入尺寸、得分阈值、非极大值抑制阈值
    this->yunet_detector = cv::FaceDetectorYN::create(yunet_model_path, "", cv::Size(640, 640), CONFIDENCE_THRESHOLD, NMS_THRESHOLD);
#else
    // 加载人脸检测器 (HOG + SVM)
    this->detector = get_frontal_face_detector();
#endif

    // 加载人脸关键点检测器
    deserialize(shape_predictor_path) >> sp;

    // 加载深度学习人脸识别模型
    deserialize(face_recognition_model_path) >> net;
}

// 人脸检测，返回人脸的方框列表
std::vector<dlib::full_object_detection> FaceCoder::detect_faces(const cv_image<rgb_pixel> &img)
{
    std::vector<dlib::full_object_detection> shapes;

#ifdef USE_YUNET_DETECTOR
    // 使用 YuNet 检测人脸

    cv_image<rgb_pixel> dlib_img = img;
    cv::Mat yunet_result;

    // 使用 dlib 提供的 toMat() 将 dlib 图片转换为 cv::Mat，然后将 RGB 转为 BGR 以供 OpenCV 模型使用
    cv::Mat mat = dlib::toMat(dlib_img);
    cv::Mat bgr_mat;
    cv::cvtColor(mat, bgr_mat, cv::COLOR_RGB2BGR);

    // 设置输入图像的size
    const cv::Size input_size = bgr_mat.size();
    // 如果跟上一个图像的分辨率一样的话就不需要去再次设置了
    if (input_size.width != this->width || input_size.height != this->height)
    {
        this->yunet_detector->setInputSize(input_size);
        this->width = input_size.width;
        this->height = input_size.height;
    }

    // 人脸检测处理
    this->yunet_detector->detect(bgr_mat, yunet_result);

    for (int i = 0; i < yunet_result.rows; i++)
    {
        float x = yunet_result.at<float>(i, 0);
        float y = yunet_result.at<float>(i, 1);
        float w = yunet_result.at<float>(i, 2);
        float h = yunet_result.at<float>(i, 3);
        // 1. 构造 dlib 矩形框
        dlib::rectangle rect(
            static_cast<long>(x),
            static_cast<long>(y),
            static_cast<long>(x + w),
            static_cast<long>(y + h)
        );

        // 2. 提取 5 个关键点并填入 dlib 格式
        dlib::full_object_detection shape = this->sp(img, rect);
        shapes.push_back(shape);
    }
#else
    // 使用 dlib 的 HOG + SVM 检测所有人脸
    auto faces = this->detector(img);
    // 获取人脸特征68/5点
    for(auto& face : faces)
    {
        dlib::full_object_detection shape = this->sp(img, face);
        shapes.push_back(shape);
    }
#endif

    return shapes;
}



// 图像转换为人脸数据结构
std::vector<FaceData> FaceCoder::img_to_facedata(const cv_image<rgb_pixel> &img, const std::string &name)
{
    std::vector<FaceData> face_datas;

    auto shapes = this->detect_faces(img);

    // std::cout << "检测到 " << faces.size() << " 张人脸" << std::endl;

    // 2. 为每张人脸提取编码
    static int face_id = 0;
    for (const auto &shape : shapes)
    {
        try
        {
            // 裁剪并标准化人脸 (150x150, 适当填充)
            matrix<rgb_pixel> face_chip;

            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);

            // 提取128维特征向量 (人脸编码)
            matrix<float, 0, 1> encoding = this->net(face_chip);

            // 构建 FaceData 结构体
            FaceData fd;
            fd.id = face_id++;
            fd.name = name;
            fd.distance = 1.0;
            fd.encoding = encoding;
            fd.face_info = shape;
            face_datas.push_back(fd);
        }
        catch (std::exception &e)
        {
            std::cerr << "⚠️ 处理人脸时出错: " << e.what() << std::endl;
        }
    }

    return face_datas;
}

// dlib图像预处理函数
matrix<rgb_pixel> FaceCoder::preprocess_image(const matrix<rgb_pixel> &img)
{
    double scale = DEFAULT_SCALE_FACTOR;
    // 然后检查是否超过最大尺寸
    if (
        img.nc() * scale > MAX_IMAGE_WIDTH ||
        img.nr() * scale > MAX_IMAGE_HEIGHT)
    {
        double max_scale_width = static_cast<double>(MAX_IMAGE_WIDTH) / img.nc();
        double max_scale_height = static_cast<double>(MAX_IMAGE_HEIGHT) / img.nr();
        double max_allowed_scale = std::min(max_scale_width, max_scale_height);

        // 取缩放系数和最大允许缩放的最小值
        scale = std::min(scale, max_allowed_scale);
    }

    // 确保缩放比例合理
    scale = std::max(0.1, std::min(scale, 1.0)); // 限制在0.1-1.0之间
    // 计算新的尺寸
    int new_width = static_cast<int>(img.nc() * scale);
    int new_height = static_cast<int>(img.nr() * scale);

    matrix<rgb_pixel> resized_img;
    // 为避免与 resize_image 的另一个模板重载发生歧义，先设置输出图像的尺寸，
    // 然后使用带插值器的重载：resize_image(const image_type& in, image_type& out, const interp&)
    // 这样不会把第三个参数误推断为插值器类型。
    resized_img.set_size(new_height, new_width);
    dlib::resize_image(img, resized_img, dlib::interpolate_bilinear());

    this->scale = scale;
    return resized_img;
}

// OpenCV图像预处理
cv::Mat FaceCoder::preprocess_image(const cv::Mat &cv_img)
{
    double scale = DEFAULT_SCALE_FACTOR;

    // 然后检查是否超过最大尺寸
    if (
        cv_img.cols * scale > MAX_IMAGE_WIDTH ||
        cv_img.rows * scale > MAX_IMAGE_HEIGHT)
    {

        double max_scale_width = static_cast<double>(MAX_IMAGE_WIDTH) / cv_img.cols;
        double max_scale_height = static_cast<double>(MAX_IMAGE_HEIGHT) / cv_img.rows;
        double max_allowed_scale = std::min(max_scale_width, max_scale_height);

        // 取缩放系数和最大允许缩放的最小值
        scale = std::min(scale, max_allowed_scale);
    }

    // 确保缩放比例合理
    scale = std::max(0.1, std::min(scale, 1.0)); // 限制在0.1-1.0之间

    // 计算新的尺寸
    int new_width = static_cast<int>(cv_img.cols * scale);
    int new_height = static_cast<int>(cv_img.rows * scale);

    cv::Mat resized_img, rgb_img;
    cv::resize(cv_img, resized_img, cv::Size(new_width, new_height));

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

    this->scale = scale;
    return rgb_img;
}

// 从图像文件获取所有人脸数据
std::vector<FaceData> FaceCoder::get_face_data(const std::string &image_path, const std::string &name)
{
    cv::Mat img = cv::imread(image_path);
    if (img.empty())
    {
        std::cerr << "无法加载图像: " << image_path << std::endl;
        return std::vector<FaceData>();
    }

    cv::Mat rgb_img = preprocess_image(img);

    return img_to_facedata(rgb_img, name);
}

// 从 OpenCV 图像获取所有人脸数据
std::vector<FaceData> FaceCoder::get_face_data(const cv::Mat &cv_img, const std::string &name)
{
    cv::Mat rgb_img;

    // 预处理 OpenCV 图像
    rgb_img = preprocess_image(cv_img);

    // 将 cv::Mat 包装成 dlib 格式（这里是浅拷贝，不产生额外开销）
    cv_image<rgb_pixel> dlib_img(rgb_img);

    return img_to_facedata(dlib_img, name);
}

// 已有人脸库与单个人脸进行比较，判断是否匹配 (默认阈值0.6),输入图片中可能有多张人脸，多个人脸全部匹配成功才返回True
bool FaceCoder::compare_faces(const std::vector<FaceData> &known_facedatas,
                              std::vector<FaceData> &face_datas_to_check,
                              double tolerance)
{
    bool find_match = false;
    for (int i = 0; i < face_datas_to_check.size(); i++)
    {
        find_match = false;
        for (const auto &known_facedata : known_facedatas)
        {
            // 计算欧氏距离 (与Python face_recognition库相同)
            double distance = dlib::length(known_facedata.encoding - face_datas_to_check[i].encoding);
            if (distance <= tolerance)
            {
                find_match = true;
                face_datas_to_check[i].name = known_facedata.name;
                face_datas_to_check[i].distance = distance;
            }
        }
        if (!find_match)
        {
            face_datas_to_check[i].name = "unknown";
            std::cout << "未识别出匹配的人脸，标记为 unknown" << std::endl;
        }
    }
    return find_match;
}

// 绘制人脸框,opencv格式
void FaceCoder::DrawRectangle(cv::Mat &img, std::vector<FaceData> face_datas)
{
    // std::cout << "缩放系数" << this->scale << std::endl;
    for (const auto &fd : face_datas)
    {
        int left = fd.get_rect().left() / this->scale;
        int top = fd.get_rect().top() / this->scale;
        int right = fd.get_rect().right() / this->scale;
        int bottom = fd.get_rect().bottom() / this->scale;
        cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 3);
        cv::putText(img, fd.name, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
}

FaceCoder::~FaceCoder()
{
    // 析构函数，释放资源（如果有需要）
}