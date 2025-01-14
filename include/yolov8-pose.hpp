//
// Created by ubuntu on 1/20/23.
//
#ifndef JETSON_POSE_YOLOV8_POSE_HPP
#define JETSON_POSE_YOLOV8_POSE_HPP

#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"
#include "obj.hpp"

using namespace pose;

class YOLOv8_pose {
public:
    explicit YOLOv8_pose(const std::string& engine_file_path);

    ~YOLOv8_pose();

    void make_pipe(bool warmup = true);

    void copy_from_Mat(const cv::Mat& image);

    void copy_from_Mat(const cv::Mat& image, cv::Size& size);

    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

    void infer();

    void postprocess(std::vector<Object>& objs, float score_thres = 0.25f, float iou_thres = 0.65f, int topk = 100);

    static void draw_objects(const cv::Mat&                                image,
                             cv::Mat&                                      res,
                             const std::vector<Object>&                    objs,
                             const std::vector<std::vector<unsigned int>>& SKELETON,
                             const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                             const std::vector<std::vector<unsigned int>>& LIMB_COLORS,
                             Tracker &Tk); 

    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    PreParam pparam;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};

YOLOv8_pose::YOLOv8_pose(const std::string& engine_file_path)
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbBindings();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            dims         = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
        }
        else {
            dims         = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8_pose::~YOLOv8_pose()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void YOLOv8_pose::make_pipe(bool warmup)
{

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMalloc(&d_ptr, bindings.size * bindings.dsize));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMalloc(&d_ptr, size));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8_pose::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}

void YOLOv8_pose::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    auto     width      = in_binding.dims.d[3];
    auto     height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_pose::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_pose::infer()
{

    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void YOLOv8_pose::postprocess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];
    // std::cout << num_channels << std::endl;
    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::Rect>           bboxes;
    std::vector<float>              scores;
    std::vector<int>                labels;
    std::vector<int>                indices;
    std::vector<std::vector<float>> kpss;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr    = output.row(i).ptr<float>();
        // 18 = 4 box, 4 label, 5*2 points
        auto bboxes_ptr = row_ptr;  
        auto scores_ptr = row_ptr + 4;
        auto kps_ptr    = row_ptr + 8;

        float score = *scores_ptr;
        
        for(int j = 0; j < 4; j++, scores_ptr++)
        {
            float score = *scores_ptr;
            if (score > score_thres) 
            {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;
            std::vector<float> kps;
            for (int k = 0; k < 5; k++) {
                float kps_x = (*(kps_ptr + 2 * k) - dw) * ratio;
                float kps_y = (*(kps_ptr + 2 * k + 1) - dh) * ratio;
                // float kps_s = *(kps_ptr + 3 * k + 2);
                kps_x       = clamp(kps_x, 0.f, width);
                kps_y       = clamp(kps_y, 0.f, height);
                kps.push_back(kps_x);
                kps.push_back(kps_y);
                // kps.push_back(kps_s);
            }

            bboxes.push_back(bbox);
            labels.push_back(j);
            scores.push_back(score);
            kpss.push_back(kps);
            }
        }
        
    }
// cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
// #define BATCHED_NMS
#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        obj.kps   = kpss[i];
        objs.push_back(obj);
        cnt += 1;
    }
}

void YOLOv8_pose::draw_objects(const cv::Mat&                                image,
                               cv::Mat&                                      res,
                               const std::vector<Object>&                    objs,
                               const std::vector<std::vector<unsigned int>>& SKELETON,
                               const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                               const std::vector<std::vector<unsigned int>>& LIMB_COLORS,
                               Tracker &Tk)// store blades
{
    res                 = image.clone();
    const int num_point = 5; // in  , 2, 3
    for (auto& obj : objs) {
        cv::rectangle(res, obj.rect, {0, 0, 255}, 2);

        char text[256];
        switch (obj.label)
        {
        case 0:
            sprintf(text, "red_hitting %.1f%%", obj.prob * 100);
            break;
        case 1:
            sprintf(text, "red_hit %.1f%%", obj.prob * 100);
            break;
        case 2:
            sprintf(text, "blue_hitting %.1f%%", obj.prob * 100);
            break;
        case 3:
            sprintf(text, "blue_hit %.1f%%", obj.prob * 100);
            break;
        default:
            sprintf(text, "Error type!");
            break;
        }
        // printf("%d\n", obj.label);
        

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);

        auto& kps = obj.kps;
        if(obj.label==0||obj.label==2)
        {
            Tk.getdate(obj.rect, kps);
            Tk.draw_res(res, obj.rect, kps);
        }
        
        // printf("%f\t%f\t%f\n", kps[0], kps[1], kps[2]);
        // for (int k = 0; k < num_point; k++) {
            
        //     int   kps_x = std::round(kps[k * 2]);
        //     int   kps_y = std::round(kps[k * 2 + 1]);
        //     // float kps_s = kps[k * 3 + 2];
        //     // if (kps_s > 0.5f) {
        //     cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
        //     cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
        //     //     // printf("%f\t%f\t%f\n", kps[0], kps[1], kps[2]);
        //     // }
        //     // if(k < 2)// draw lines
        //     // {
        //     //     auto& ske    = SKELETON[k];
        //     //     int   pos1_x = std::round(kps[(ske[0] ) * 2]);
        //     //     int   pos1_y = std::round(kps[(ske[0] ) * 2 + 1]);

        //     //     int pos2_x = std::round(kps[(ske[1] ) * 2]);
        //     //     int pos2_y = std::round(kps[(ske[1] ) * 2 + 1]);
        //     //     cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
        //     //     cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
        //     // }
        //     // auto& ske    = SKELETON[k];
        //     // int   pos1_x = std::round(kps[(ske[0] ) * 2]);
        //     // int   pos1_y = std::round(kps[(ske[0] ) * 2 + 1]);

        //     // int pos2_x = std::round(kps[(ske[1] ) * 2]);
        //     // int pos2_y = std::round(kps[(ske[1] ) * 2 + 1]);

        //     // // float pos1_s = kps[(ske[0] ) * 3 + 2];
        //     // // float pos2_s = kps[(ske[1] ) * 3 + 2];

        //     // // if (pos1_s > 0.5f && pos2_s > 0.5f) {
        //     //     cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
        //     //     cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
        //     // // }
        // }
    }
}

#endif  // JETSON_POSE_YOLOV8_POSE_HPP
