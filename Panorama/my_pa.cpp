/*
 * @file my_pa.cpp
 * @author Jessie Peng
 */

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "my_pa.h"

using namespace std;
using namespace cv;

bool Panoramaxxxx::makePanorama(vector<Mat>& img_vec, Mat& img_out, double f)
{
    int n = img_vec.size();

    /* Step 1: Warp the images into cylindrical coordinates */

    vector<Mat> img_cylinder;
    for (auto img : img_vec)
    {
//        imshow("img", warpCylinder(img, f, f));
//        waitKey(0);
        img_cylinder.push_back(warpCylinder(img, f, f)); // set r = f
    }
    destroyAllWindows();

    /* Step 2: Detect features with SIFT */

    // note the OpenCV 3.X standard of using detectors
    Ptr<xfeatures2d::SiftFeatureDetector> sift = xfeatures2d::SiftFeatureDetector::create();
    vector<KeyPoint> keypoint[n];
    Mat img_keypoint[n];
    for (int i = 0; i < n; i++)
    {
        sift->detect(img_cylinder[i], keypoint[i]);
        cout << keypoint[i].size() << endl;
        drawKeypoints(img_cylinder[i], keypoint[i], img_keypoint[i]);
        imshow("kp", img_keypoint[i]);
        waitKey(0);
    }
    destroyAllWindows();

    // Step 3: Image registration

    // Step 4: Image blending

    img_out = img_vec[0];
    return false;
}

Mat Panoramaxxxx::warpCylinder(Mat img, double r, double f)
{
    int x_range = (img.cols - 1) / 2;
    int y_range = (img.rows - 1) / 2;
    int ret_x_range = static_cast<int>(plane_to_cylinder_X(x_range, r, f));
    int ret_y_range = static_cast<int>(plane_to_cylinder_Y(0, y_range, r, f)); // when x=0, y is the maximum
    int ret_cols = ret_x_range * 2 + 1;
    int ret_rows = ret_y_range * 2 + 1;

    Mat img_ret = Mat::zeros(ret_rows, ret_cols, CV_8UC3);
    for (int yy = 0; yy < ret_rows; yy++)
    {
        for (int xx = 0; xx < ret_cols; xx++)
        {
            int x = cvRound(cylinder_to_plane_X(xx - ret_x_range, r, f));
            int y = cvRound(cylinder_to_plane_Y(x, yy - ret_y_range, r, f));
            x += x_range;
            y += y_range;

            if (x >= 0 && x <= img.cols && y >= 0 && y <= img.rows)
            {
                img_ret.at<Vec3b>(yy, xx) = img.at<Vec3b>(y, x);
            }
        }
    }

    return img_ret;
}

double Panoramaxxxx::plane_to_cylinder_X(double x, double r, double f)
{
    return r * atan(x / f);
}

double Panoramaxxxx::plane_to_cylinder_Y(double x, double y, double r, double f)
{
    return r * y / sqrt(x * x + f * f);
}

double Panoramaxxxx::cylinder_to_plane_X(double xx, double r, double f)
{
    return f * tan(xx / r);
}

double Panoramaxxxx::cylinder_to_plane_Y(double x, double yy, double r, double f)
{
    return yy / r * sqrt(x * x + f * f);
}