//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//计时器函数的封装类，可自选精度，有新需求回头修改
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _TIMER_H_
#define _TIMER_H_

#include <iostream>
#include <chrono>
#include <ctime>
#include <time.h>

enum MODEL{SECOND,MILLISECOND,MICROSECOND,NANOSECOND};

using namespace std::chrono;

class Timer
{
public:
    Timer();
    void begin();
    void close();
    void end();
    bool status();
    double thisTime();            //返回当前时刻相对于计时器创建时用时多少秒
    double timeConsuming(int model=MILLISECOND);
    time_t getSystemTime();
private:
    system_clock::time_point time_begin;
    system_clock::time_point time_end;
    double timeConsume;
    bool isWorking;                                 //是否在计时
    system_clock::time_point init_time;             //记录计时器创建时刻                                                
};
#endif