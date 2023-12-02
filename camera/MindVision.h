//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//迈徳威视工业相机函数的封装，回头补充新功能
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _MINDVISION_H_
#define _MINDVISION_H_
// #include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include "MVSDK/CameraApi.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "../assert/Timer/Timer.h"
#include <iostream>
#include <string>

class MindVision {
public:
    MindVision();
    
    void open();

    void getImage(cv::Mat & src);

    void close();
    
    ~MindVision();
    
private:
    void adjustParam();
private:
    int                     iStatus=-1;
    int                     hCamera;
    int                     iCameraCounts=1;
    int                     channel=3;
    tSdkCameraDevInfo       cameraInform;
    tSdkCameraCapbility     cameraSettings;
    tSdkFrameHead           frameHead;
    BYTE*                   pbyBuffer=NULL;
    int                     exposure=10;                      //单位毫秒
    char*                   load="../camera/Data/bubing.config";
    int                     gamma=100;
    Timer                   clock;
};

#endif