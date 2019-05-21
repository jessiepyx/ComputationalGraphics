/**
 * @file my_pa.h
 * @author Jessie Peng
 */

#ifndef PANORAMA_MY_PA_H
#define PANORAMA_MY_PA_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "hw6_pa.h"

using namespace std;
using namespace cv;

class Panoramaxxxx: public CylindricalPanorama {
public:
    bool makePanorama(vector<Mat>& img_vec, Mat& img_out, double f);

private:
    Mat warpCylinder(Mat& img, double r, double f);
    double plane_to_cylinder_X(double x, double r, double f);
    double plane_to_cylinder_Y(double x, double y, double r, double f);
    double cylinder_to_plane_X(double x, double r, double f);
    double cylinder_to_plane_Y(double x, double y, double r, double f);
    void registerImage(Mat& src, Mat& dst, Mat& out, Mat& src_mask, Mat& out_mask);
    void blendImage(vector<Mat>& img_vec, Mat& img_out, vector<Mat>& mask);

    int n; // number of input images
    vector<Mat> img_cylinder; // input images in cylindrical coordinates
    Mat mask_cylinder;
    vector<Mat> img_rectified; // images after transformation according to homography
    vector<Mat> mask_rectified;
};

#endif //PANORAMA_MY_PA_H
