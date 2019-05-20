/**
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
    if (n <= 0)
    {
        return false;
    }

    /** Warp the images into cylindrical coordinates */
    vector<Mat> img_cylinder;
    for (auto img : img_vec)
    {
        img_cylinder.push_back(warpCylinder(img, f, f)); // set r = f
    }


    /** Stitch images */
    int start = n / 2; // start from the middle
    Mat img_start = img_cylinder[start];
    img_start.copyTo(img_out(
            Rect((img_out.cols - img_start.cols) / 2, img_out.rows - img_start.rows, img_start.cols, img_start.rows)
            )); // copy first image to the lower middle

    Mat img_dst = Mat::zeros(img_out.size(), CV_8UC3);
    img_start.copyTo(img_dst(
            Rect((img_out.cols - img_start.cols) / 2, img_out.rows - img_start.rows, img_start.cols, img_start.rows)
            ));
    for (int i = start + 1; i < n; i++) // to right
    {
        stitchTwoImages(img_cylinder[i], img_dst, img_out);
    }

    img_start.copyTo(img_dst(Rect((img_out.cols - img_start.cols) / 2, img_out.rows - img_start.rows, img_start.cols, img_start.rows)));
    for (int i = start - 1; i >= 0; i--) // to left
    {
        stitchTwoImages(img_cylinder[i], img_dst, img_out);
    }

    return true;
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

void Panoramaxxxx::stitchTwoImages(Mat& src, Mat& dst, Mat& out)
{
    /** Step 1: Detect features*/

    // note the OpenCV 3.X standard
    Ptr<xfeatures2d::SiftFeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create();
    Ptr<xfeatures2d::SiftDescriptorExtractor> extractor = xfeatures2d::SiftDescriptorExtractor::create();

    vector<KeyPoint> keypoint_src, keypoint_dst;
    Mat img_keypoint_src, img_keypoint_dst;
    Mat descriptor_src, descriptor_dst;

    /// Detect keypoints with SIFT
    detector->detect(src, keypoint_src);

//    drawKeypoints(src, keypoint_src, img_keypoint_src);
//    imshow("keypoints in src", img_keypoint_src);
//    waitKey(0);

    detector->detect(dst, keypoint_dst);

//    drawKeypoints(dst, keypoint_dst, img_keypoint_dst);
//    imshow("keypoints in dst", img_keypoint_dst);
//    waitKey(0);

    /// Extract descriptors
    extractor->compute(src, keypoint_src, descriptor_src);

//    imshow("descriptors in src", descriptor_src);
//    waitKey(0);

    extractor->compute(dst, keypoint_dst, descriptor_dst);

//    imshow("descriptors in dst", descriptor_dst);
//    waitKey(0);
//    destroyAllWindows();

    /** Step 2: Match features*/

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");

    vector<DMatch> matches;
    vector<DMatch> good_matches;
    vector<DMatch> inline_matches;
    Mat img_match;
    int good_match_size = 500;

    /// Compute all feature correspondences with brute force
    matcher->match(descriptor_dst, descriptor_src, matches);

//    drawMatches(dst, keypoint_dst, src, keypoint_src, matches, img_match);
//    imshow("matches", img_match);
//    waitKey(0);
//    destroyAllWindows();

    /// Keep the top 500 matches
    Mat dist = Mat::zeros(matches.size(), 1, CV_32F);
    Mat sorted_ind;
    for (int i = 0; i < matches.size(); i++)
    {
        dist.at<float>(i, 0) = matches[i].distance;
    }
    sortIdx(dist, sorted_ind, SORT_EVERY_COLUMN + SORT_ASCENDING);

    for (int i = 0; i < good_match_size; i++)
    {
        good_matches.push_back(matches[sorted_ind.at<int>(i, 0)]);
    }

//    drawMatches(dst, keypoint_dst, src, keypoint_src, good_matches, img_match);
//    imshow("good matches", img_match);
//    waitKey(0);
//    destroyAllWindows();

    /// Eliminate outliers with RANSAC
    vector<Point2f> src_float, dst_float;
    for (auto it = good_matches.begin(); it != good_matches.end(); it++)
    {
        src_float.push_back(keypoint_src[it->trainIdx].pt);
        dst_float.push_back(keypoint_dst[it->queryIdx].pt);
    }

    vector<uchar> is_inliners(good_match_size);
    Mat homography = findHomography(src_float, dst_float, CV_RANSAC, 3, is_inliners);

    for (int i = 0; i < is_inliners.size(); i++)
    {
        if (is_inliners[i])
        {
            inline_matches.push_back(good_matches[i]);
        }
    }

//    drawMatches(dst, keypoint_dst, src, keypoint_src, inline_matches, img_match);
//    imshow("inline matches", img_match);
//    waitKey(0);
//    destroyAllWindows();

    /** Step 3: Register images */

    Mat src_pers(dst.size(), CV_8UC3);
    warpPerspective(src, src_pers, homography, dst.size());

    /** Step 4: Blend images */
    for (int j = 0; j < dst.rows; j++)
    {
        for (int i = 0; i < dst.cols; i++)
        {
            if (norm(out.at<Vec3b>(j, i)) == 0)
            {
                out.at<Vec3b>(j, i) = src_pers.at<Vec3b>(j, i);
            }
        }
    }

    /** Step 5: Warp the stitched image into plane coordinates */

//    imshow("stitched", out);
//    waitKey(0);
//    destroyAllWindows();

    dst = src_pers;
}