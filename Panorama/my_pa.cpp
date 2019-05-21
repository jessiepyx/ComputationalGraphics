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
    n = img_vec.size();
    if (n <= 0)
    {
        return false;
    }

    /** Part 1
     *  Warp the images into cylindrical coordinates
     */
    for (auto img : img_vec)
    {
        img_cylinder.push_back(warpCylinder(img, f, f)); // set r = f
    }

    Mat blank = Mat::ones(img_vec[0].size(), CV_8UC3);
    blank *= 255;
    mask_cylinder = warpCylinder(blank, f, f);

    /** Part 2
     *  Image registration
     */
    for (int i = 0; i < n; i++)
    {
        img_rectified.push_back(Mat::zeros(img_out.size(), CV_8UC3));
        mask_rectified.push_back(Mat::zeros(img_out.size(), CV_8UC3));
    }

    /** take the middle image as reference */
    int ref = n / 2;
    Mat img_ref = img_cylinder[ref];
    img_ref.copyTo(img_rectified[ref](Rect(
            (img_out.cols - img_ref.cols) / 2, (img_out.rows - img_ref.rows) / 2, img_ref.cols, img_ref.rows
            ))); // put the reference image in the center

    mask_cylinder.copyTo(mask_rectified[ref](Rect(
            (img_out.cols - img_ref.cols) / 2, (img_out.rows - img_ref.rows) / 2, img_ref.cols, img_ref.rows
    )));

    /** register images on the right side */
    for (int i = ref; i < n - 1; i++)
    {
        registerImage(img_cylinder[i+1], img_rectified[i], img_rectified[i+1], mask_cylinder, mask_rectified[i+1]);
    }

    /** register images on the left side */
    for (int i = ref; i >= 1; i--)
    {
        registerImage(img_cylinder[i-1], img_rectified[i], img_rectified[i-1], mask_cylinder, mask_rectified[i-1]);
    }

    /** Part 3
     *  Image blending
     */
    blendImage(img_rectified, img_out, mask_rectified);

    return true;
}

Mat Panoramaxxxx::warpCylinder(Mat& img, double r, double f)
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

void Panoramaxxxx::registerImage(Mat& src, Mat& dst, Mat& out, Mat& src_mask, Mat& out_mask)
{
    /** Step 1
     *  Detect features
     */
    // note the OpenCV 3.X standard
    Ptr<xfeatures2d::SiftFeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create();
    Ptr<xfeatures2d::SiftDescriptorExtractor> extractor = xfeatures2d::SiftDescriptorExtractor::create();

    vector<KeyPoint> keypoint_src, keypoint_dst;
    Mat img_keypoint_src, img_keypoint_dst;
    Mat descriptor_src, descriptor_dst;

    /** 1.1 Detect keypoints with SIFT */
    detector->detect(src, keypoint_src);

//    drawKeypoints(src, keypoint_src, img_keypoint_src);
//    imshow("keypoints in src", img_keypoint_src);
//    waitKey(0);

    detector->detect(dst, keypoint_dst);

//    drawKeypoints(dst, keypoint_dst, img_keypoint_dst);
//    imshow("keypoints in dst", img_keypoint_dst);
//    waitKey(0);

    /** 1.2 Extract descriptors */
    extractor->compute(src, keypoint_src, descriptor_src);

//    imshow("descriptors in src", descriptor_src);
//    waitKey(0);

    extractor->compute(dst, keypoint_dst, descriptor_dst);

//    imshow("descriptors in dst", descriptor_dst);
//    waitKey(0);
//    destroyAllWindows();

    /** Step 2
     *  Match features
     */
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");

    vector<DMatch> matches;
    vector<DMatch> good_matches;
    vector<DMatch> inlier_matches;

    Mat img_match;
    int good_match_size = 500;

    /** 2.1 Compute all correspondences with brute force */
    matcher->match(descriptor_dst, descriptor_src, matches);

//    drawMatches(dst, keypoint_dst, src, keypoint_src, matches, img_match);
//    imshow("matches", img_match);
//    waitKey(0);
//    destroyAllWindows();

    /** 2.2 Keep the top 500 matches */
    Mat dist = Mat::zeros(matches.size(), 1, CV_32F);
    Mat sorted_idx;
    for (int i = 0; i < matches.size(); i++)
    {
        dist.at<float>(i, 0) = matches[i].distance;
    }
    sortIdx(dist, sorted_idx, SORT_EVERY_COLUMN + SORT_ASCENDING);

    for (int i = 0; i < good_match_size; i++)
    {
        good_matches.push_back(matches[sorted_idx.at<int>(i, 0)]);
    }

//    drawMatches(dst, keypoint_dst, src, keypoint_src, good_matches, img_match);
//    imshow("good matches", img_match);
//    waitKey(0);
//    destroyAllWindows();

    /** 2.3 Eliminate outliers with RANSAC */
    vector<Point2f> src_float, dst_float;
    for (auto it = good_matches.begin(); it != good_matches.end(); it++)
    {
        src_float.push_back(keypoint_src[it->trainIdx].pt);
        dst_float.push_back(keypoint_dst[it->queryIdx].pt);
    }

    vector<uchar> is_inliers(good_match_size);
    Mat homography = findHomography(src_float, dst_float, CV_RANSAC, 3, is_inliers);

    for (int i = 0; i < is_inliers.size(); i++)
    {
        if (is_inliers[i])
        {
            inlier_matches.push_back(good_matches[i]);
        }
    }

//    drawMatches(dst, keypoint_dst, src, keypoint_src, inlier_matches, img_match);
//    imshow("inlier matches", img_match);
//    waitKey(0);
//    destroyAllWindows();

    src_float.clear();
    dst_float.clear();
    for (auto it = inlier_matches.begin(); it != inlier_matches.end(); it++)
    {
        src_float.push_back(keypoint_src[it->trainIdx].pt);
        dst_float.push_back(keypoint_dst[it->queryIdx].pt);
    }

    homography = findHomography(src_float, dst_float, CV_RANSAC);

    /** Step 3
     *  Register images
     */
    warpPerspective(src, out, homography, out.size(), INTER_NEAREST);
    warpPerspective(src_mask, out_mask, homography, out.size(), INTER_NEAREST);
}

void Panoramaxxxx::blendImage(vector<Mat>& img_vec, Mat& img_out, vector<Mat>& mask)
{
    vector<Vec3f> candidates;
    Vec3f sum, average;

    for (int i = 0; i < img_out.rows; i++)
    {
        for (int j = 0; j < img_out.cols; j++)
        {
            candidates.clear();
            sum = Vec3b(0, 0, 0);
            for (auto img : img_vec)
            {
                Vec3b tmp = img.at<Vec3b>(i, j);
                if (norm(tmp, NORM_L1) > 0)
                {
                    candidates.push_back(tmp);
                    sum += tmp;
                }
            }

            int cand_num = candidates.size();
            if (cand_num > 1)
            {
                average = sum / cand_num;

                /// get rid of noise
                if (cand_num > 2)
                {
                    double dist[cand_num];
                    for (int k = 0; k < cand_num; k++)
                    {
                        dist[k] = norm(candidates[k] - average, NORM_L2);
                    }

                    double max_dist = dist[0];
                    int max_idx = 0;
                    double sum_dist = dist[0];
                    for (int k = 1; k < cand_num; k++)
                    {
                        if (dist[k] > max_dist)
                        {
                            max_dist = dist[k];
                            max_idx = k;
                        }
                        sum_dist += dist[k];
                    }

                    if (max_dist > 20 && max_dist > sum_dist / cand_num * 1.5) // filter out noise
                    {
                        sum -= candidates[max_idx];
                        average = sum / (cand_num - 1);
                    }
                }

                img_out.at<Vec3b>(i, j) = average;
            }
            else if (cand_num == 1)
            {
                img_out.at<Vec3b>(i, j) = candidates[0];
            }
        }
    }
}