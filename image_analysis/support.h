/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#ifndef __SUPPORT_H__
#define __SUPPORT_H__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

#ifdef __cplusplus
}
#endif

#endif
