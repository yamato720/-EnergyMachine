//
// Created by ubuntu on 4/7/23.
//
#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8-pose.hpp"
#include "camera/MindVision.h"

const std::vector<std::vector<unsigned int>> KPS_COLORS = {{0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           };

const std::vector<std::vector<unsigned int>> SKELETON = {{1, 1},
                                                         {1, 3},
                                                         {2, 4},
                                                         {1, 4},
                                                         {2, 3}
                                                         };

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0}
                                                            };

int main(int argc, char** argv)
{
    // cuda:0
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};

    assert(argc == 2);

    auto yolov8_pose = new YOLOv8_pose(engine_file_path);
    yolov8_pose->make_pipe(true);

    cv::Mat  res, image;
    cv::Size size        = cv::Size{640, 640};
    int      topk        = 100;
    float    score_thres = 0.1f;
    float    iou_thres   = 0.50f;

    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    
    // cv::VideoCapture cap(0);
    MindVision cap;
    // if (!cap.isOpened()) {
    //     printf("can not open %s\n", path.c_str());
    //     return -1;
    // }
    while (1) {
        cap.getImage(image);
        objs.clear();
        yolov8_pose->copy_from_Mat(image, size);
        auto start = std::chrono::system_clock::now();
        yolov8_pose->infer();
        
        yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
        yolov8_pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        // printf("cost %2.4lf ms\n", tc);
        char text[256];
        sprintf(text, "cost %2.4lf ms, fps %2.4lf", tc, 1000.0/tc);
        cv::putText(res, text, cv::Point(0,25), cv::FONT_HERSHEY_SIMPLEX, 1, {255, 0, 0}, 2);
        cv::imshow("result", res);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    
    cv::destroyAllWindows();
    delete yolov8_pose;
    return 0;
}
