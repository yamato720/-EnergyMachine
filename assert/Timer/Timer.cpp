#include "Timer.h"

Timer::Timer()
{
    timeConsume=0;
    isWorking=false;
    init_time=high_resolution_clock::now();
}

void Timer::begin()
{
    time_begin=high_resolution_clock::now();
    isWorking=true;
}

void Timer::close()
{
    time_begin=high_resolution_clock::now();
    time_end=time_begin;
    isWorking=false;
}

void Timer::end()
{
    time_end=high_resolution_clock::now();
}

bool Timer::status()
{
    return isWorking;
}

double Timer::timeConsuming(int model)
{
    //如果处于工作状态返回正常值
    if (isWorking == true)
    {
        if (model == SECOND)
        {
            return duration_cast<seconds>(time_end - time_begin).count();
        }
        if (model == MILLISECOND)
        {
            return duration_cast<milliseconds>(time_end - time_begin).count();
        }
        if (model == MICROSECOND)
        {
            return duration_cast<microseconds>(time_end - time_begin).count();
        }
        if (model == NANOSECOND)
        {
            return duration_cast<nanoseconds>(time_end - time_begin).count();
        }
    }
    else
    {
        return 1000;
    }
}

time_t Timer::getSystemTime()
{
    system_clock::time_point timePoint=system_clock::now();
    time_t time_tPoint=system_clock::to_time_t(timePoint);
    
    return time_tPoint;
}

double Timer::thisTime()
{
    double time_used=0;
    system_clock::time_point current_time=system_clock::now();
    time_used=duration_cast<milliseconds>(current_time-init_time).count();
    return (double)time_used/1000.0;
}