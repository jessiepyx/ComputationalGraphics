/*
 * @file my_pa.h
 * @author Jessie Peng
 */

#ifndef PANORAMA_MY_PA_H
#define PANORAMA_MY_PA_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "hw6_pa.h"

class Panoramaxxxx: public CylindricalPanorama {
public:
    bool makePanorama(std::vector<cv::Mat>& img_vec, cv::Mat& img_out, double f);
};

#endif //PANORAMA_MY_PA_H
