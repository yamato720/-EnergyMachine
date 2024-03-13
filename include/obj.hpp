#ifndef Bobj
#include <iostream>
#include <memory.h>
#include <vector>
#include <sys/time.h>
using namespace std;

#define DISTANCE 50 // distance for two points in an interval
#define PI 3.1415926

static inline double getavg(double *a, int l)
{
    double temp = 0;
    int i = l;
    while(i--)
    {
        temp += a[i];
    }
    return temp/l;
}
class OBJ{
    public:
        OBJ(){status=0;ptr=0;full=0;empty=1;};
        bool update_xy(double x_n, double y_n);
        bool find_cur(double x_n, double y_n);
        bool getlist(double *x, double *y);


        int status; // 0 for not ready or "hit", 1 for "hitting"
        int cur_x;
        int cur_y;
    private:
        double x[5]; // x
        double xlist[5];// sorted x
        double y[5];
        double ylist[5];
        cv::Rect_<float> rect;
        int ptr;
        bool full; // array for data is full or not
        bool empty; // array for data is empty or not
        
};


class Tracker
{
    public:
        Tracker();
        void getdate(cv::Rect_<float> rect, std::vector<float> kps);
        void draw_res(cv::Mat &res, cv::Rect_<float> rect, std::vector<float> kps);
    private:
        vector<OBJ> blade;
        double start; // time when start
        double last_visit;
        int status; // 0 is not ready, 1 is having generated all blades
        int c_ptr; // ptr for R
        int b_ptr; // ptr for current blades
        bool c_full; // whether R points list is full or not
        int size_x;
        int size_y; // frame size;
        
        double cx;
        double cy;
        double cxlist[5];
        double cylist[5];
        struct timeval time;
        void update_center(double cx, double cy);
        void rollback();
        void first_input(cv::Rect_<float> rect, std::vector<float> kps);// rect, 5 points list
        void after_that(cv::Rect_<float> rect, std::vector<float> kps);// rect, 5 points list
};

Tracker::Tracker()
{
    for(int i=0;i<5;i++)
    {
        OBJ bd;
        blade.push_back(bd);
    }
    gettimeofday(&time, NULL);
    start = (time.tv_sec*1000+time.tv_sec/1000.0);
    status = 1; 
    c_ptr = 0;
    c_full = 0;
    b_ptr = 0;
    last_visit = 0;
}

bool OBJ::getlist(double *x, double *y)
{
    if(!full){
        x=NULL, y=NULL;
        return 0;
    }
    for(int i=0, temp=ptr; i<5; i++)
    {
        xlist[i]=this->x[temp];
        ylist[i]=this->y[temp];
        temp = (temp+1) % 5;
    }
    memcpy(x, xlist, 5*sizeof(double));
    memcpy(y, ylist, 5*sizeof(double));
    return 1;
}

bool OBJ::update_xy(double x_n, double y_n)
{
    if(empty)
    {
        empty = 0;
        x[0] = x_n, y[0] = y_n;
        ptr=ptr+1;
        // printf("success1!\n");
        cur_x = int(x_n);
        cur_y = int(y_n);
        return 1;
    }
    int ptr0 = ptr-1<0? 4:ptr-1; // last data ptr

    x[ptr] = x_n;
    y[ptr] = y_n;
    // printf("success2!%d %d %d\n", x[ptr], y[ptr], ptr);
    if(ptr==4)full=1;
    ptr=(ptr+1)%5;

    cur_x = int(x_n);
    cur_y = int(y_n);
    return 1;
    
}
bool OBJ::find_cur(double x_n, double y_n)
{
    int ptr0 = ptr-1<0? 4:ptr-1; // last data ptr

    if( (x_n-x[ptr0] > -DISTANCE && x_n-x[ptr0] < DISTANCE) &&
        (y_n-y[ptr0] > -DISTANCE && y_n-y[ptr0] < DISTANCE)
    ){
        status = 1;
        return 1;
    }
    status = 0;
    
    return 0;
}


void Tracker::draw_res(cv::Mat &res, cv::Rect_<float> rect, std::vector<float> kps)
{
    cv::Point C_R(std::round(cx), std::round(cy));
    cv::circle(res, C_R, 5, cv::Scalar(128, 128, 255), -1);
    for(int i=0;i<5;i++)
    {
        cv::Point end(std::round(blade[i].cur_x), std::round(blade[i].cur_y));
        if(blade[i].status)cv::line(res, C_R, end, cv::Scalar(0, 0, 255), 3);
        else cv::line(res, C_R, end, cv::Scalar(255, 0, 0), 3);
        char text[2];
        sprintf(text, "%d", i);
        cv::putText(res, text, end, cv::FONT_HERSHEY_SIMPLEX, 1, {255, 255, 255}, 3);
    }

}

void Tracker::update_center(double icx, double icy)
{
    if(c_ptr==5){c_ptr = 0;c_full = 1;}
    cxlist[c_ptr] = icx;
    cylist[c_ptr] = icy;
    if(c_full)
    {
        cx = getavg(cxlist, 5);
        cy = getavg(cylist, 5);
    }
    else
    {
        cx = getavg(cxlist, c_ptr+1);
        cy = getavg(cylist, c_ptr+1);
    }
    c_ptr++;

}
void Tracker::after_that(cv::Rect_<float> rect, std::vector<float> kps)
{
    double b1x = int(rect.x+0.5*rect.width);
    double b1y = int(rect.y+0.5*rect.height);
    double c1x = kps[2*2];
    double c1y = kps[2*2+1];

    double fbx = b1x;
    double fby = b1y;
    double fcx = c1x;
    double fcy = c1y;
    update_center(fcx, fcy);
}
void Tracker::first_input(cv::Rect_<float> rect, std::vector<float> kps)
{   // b for blade center
    // c for R
    // 1, 2 for possibility of points
    double b1x = int(rect.x+0.5*rect.width);
    double b1y = int(rect.y+0.5*rect.height);
    double c1x = kps[2*2];
    double c1y = kps[2*2+1];

    double fbx = b1x;
    double fby = b1y;
    double fcx = c1x;
    double fcy = c1y;
    update_center(fcx, fcy);


    // generate all possible position
    blade[0].update_xy(b1x, b1y);
    blade[0].status = 1;
    double x=fbx, y=fby;
    double angle = 72.0 *PI/180;
    for(int i=1;i<5;i++)
    {
        double tempx = x;
        double tempy = y;
        x = (tempx - fcx) * cos(angle) - (tempy - fcy) * sin(angle) + fcx;
        y = (tempx - fcx) * sin(angle) + (tempy - fcy) * cos(angle) + fcy;
        blade[i].update_xy(x, y);
    }


    // reupdate_center();
}


void Tracker::rollback()
{
    while(!blade.empty())blade.pop_back();
    for(int i=0;i<5;i++)
    {
        OBJ bd;
        blade.push_back(bd);
    }
    gettimeofday(&time, NULL);
    start = (time.tv_sec*1000+time.tv_sec/1000.0);
    status = 1; 
    c_ptr = 0;
    c_full = 0;
    b_ptr = 0;
    last_visit = 0;
}
void Tracker::getdate(cv::Rect_<float> rect, std::vector<float> kps)
{
    // update_center(cx, cy);
    gettimeofday(&time, NULL);
    double cur_time = (time.tv_sec*1000+time.tv_sec/1000.0);
    if(cur_time - start > 100)status = 0; // long time no see 
    switch (status)
    {
    case 0: // restart
        rollback();
        first_input(rect, kps);
        status = 1;
        break;
    
    case 1:// first
        first_input(rect, kps);
        // after_that(rect, kps);
        break;
    }
    return ;
}

#define Bobj
#endif