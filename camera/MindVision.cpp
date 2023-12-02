#include "MindVision.h"

MindVision::MindVision()
{

    CameraSdkInit(1);
    iStatus = CameraEnumerateDevice(&cameraInform, &iCameraCounts);

    std::cout << "iStatus=" << iStatus << std::endl;
    std::cout << "iCameraCount=" << iCameraCounts << std::endl;

    if (iCameraCounts == 0)
    {
        return;
    }
    iStatus = CameraInit(&cameraInform, -1, -1, &hCamera);
    if (iStatus != CAMERA_STATUS_SUCCESS)
    {
        return;
    }
    std::cout << "MindVision init success" << std::endl;
    adjustParam();
    this->open();
}

void MindVision::open()
{

    //从配置文件读取参数
    if (CameraReadParameterFromFile(hCamera, load) != CAMERA_STATUS_SUCCESS)
    {
        std::cout << "读取相机配置文件失败" << std::endl;
    }

    CameraGetCapability(hCamera, &cameraSettings);

    // g_pRgbBuffer=new unsigned char(cameraSettings.sResolutionRange.iHeightMax*cameraSettings.sResolutionRange.iWidthMax*3);

    CameraPlay(hCamera);

    CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);
}

void MindVision::getImage(cv::Mat &src)
{

    // CameraSetExposureTime(hCamera,exposure*1000);
    // CameraSetGamma(hCamera,gamma);
   
    //double exposuretime;
    //CameraGetExposureTime(hCamera, &exposuretime);
    
    //std::cout << "曝光时间：" << exposuretime << std::endl;

    if (CameraGetImageBuffer(hCamera, &frameHead, &pbyBuffer, 100) == CAMERA_STATUS_SUCCESS)
    {
        src = cv::Mat(frameHead.iHeight, frameHead.iWidth, CV_8UC3);

        CameraImageProcess(hCamera, pbyBuffer, src.data, &frameHead);

        CameraReleaseImageBuffer(hCamera, pbyBuffer);
    }
    else
    {
        std::cout << "MindVision::getImage() error" << std::endl;
    }


}

void MindVision::close()
{
    CameraUnInit(hCamera);
}

MindVision::~MindVision()
{
    close();
}

void MindVision::adjustParam()
{
    cv::namedWindow("Camera Settings", 1);
    cv::createTrackbar("gamma:", "Camera Settings", &gamma, 300, NULL);
    cv::createTrackbar("exposure:", "Camera Settings", &exposure, 100, NULL);
}