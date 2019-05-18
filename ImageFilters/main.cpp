/************************************************************************************************************
* Author: Jessie Peng
* Date: 2019-03-11
* Description: Implementation of basic image filters
*************************************************************************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <vector>

using namespace std;
using namespace cv;

/// Average Filter
void BoxFilter(const Mat &src, Mat &dst, int w, int h)
{
    // Kernel size = (2w+1) x (2h+1)
    Mat kernel = Mat::ones(w * 2 + 1, h * 2 + 1, CV_64F);
    // Normalization
    kernel /= kernel.rows * kernel.cols;
    // Convolution
    filter2D(src, dst, -1, kernel);
}

/// Gaussian Filter
void GaussianFilter(const Mat &src, Mat &dst, float sigma)
{
    // Kernel size = (2[5 sigma]+1) x (2[5 sigma]+1)
    int a = cvFloor(5 * sigma);
    Mat kernel(2 * a + 1, 2 * a + 1, CV_64F);
    for (int i = -a; i <= a; i++)
    {
        for (int j = -a; j <= a; j++)
        {
            kernel.at<double>(i+a, j+a) = exp( -(i * i + j * j) / (2 * sigma * sigma) );
        }
    }
    // Normalization:
    // Since 5 times sigma is large enough to cover most area in Gaussian distribution
    // Normalization coefficient can be estimated by 1 / (2 PI sigma^2)
    kernel /= 2 * CV_PI * sigma * sigma;
    // Convolution
    filter2D(src, dst, -1, kernel);
}

/// Median Filter
void MedianFilter(const Mat &src, Mat &dst, int w, int h)
{
    int channels = src.channels();
    // 3 channels
    if (channels == 3)
    {
        // Splits 3 channels
        vector<Mat> splitted(3), tomerge(3);
        split(src, splitted);
        for (int i = 0; i < 3; i++)
        {
            // Filters each channel respectively
            MedianFilter(splitted[i], tomerge[i], w, h);
        }
        // Merges 3 channels
        merge(tomerge, dst);
    }
    // Single channel
    else if (channels == 1)
    {
        Mat img;
        // Converts to float
        src.convertTo(img, CV_64F);
        dst = img.clone();
        // Extends borders by replication
        // Kernel size = (2w+1) x (2h+1)
        copyMakeBorder(img, img, h, h, w, w, BORDER_REPLICATE);
        vector<double> arr;
        for (int i = 0; i < dst.rows; i++)
        {
            for (int j = 0; j < dst.cols; j++)
            {
                arr.clear();
                for (int x = i; x < i + 2 * h + 1; x++)
                {
                    for (int y = j; y < j + 2 * w + 1; y++)
                    {
                        arr.push_back(img.at<double>(x, y));
                    }
                }
                // Calculates median
                sort(arr.begin(), arr.end());
                dst.at<double>(i, j) = arr[arr.size() / 2];
            }
        }
        // Convert to int
        dst.convertTo(dst, CV_8U);
    }
}

/// Bilateral Filter
void BilateralFilter(const Mat &src, Mat &dst, float sigma_s, float sigma_r)
{
    // Kernel size = (2[5m]+1) x (2[5m]+1), where m = max{sigma_s, sigma_r}
    int m = sigma_s > sigma_r ? cvFloor(5 * sigma_s) : cvFloor(5 * sigma_r);
    int channels = src.channels();
    Mat img;
    // Single channel
    if (channels == 1)
    {
        // Converts to float
        src.convertTo(img, CV_64F);
        dst = img.clone();
        // Extends borders by reflection
        int n = 2 * m + 1;
        copyMakeBorder(img, img, m, m, m, m, BORDER_REFLECT);
        double arr, weight, w;
        for (int i = 0; i < dst.rows; i++)
        {
            for (int j = 0; j < dst.cols; j++)
            {
                arr = 0;
                weight = 0;
                for (int x = i; x < i + n; x++)
                {
                    for (int y = j; y < j + n; y++)
                    {
                        // Spatial difference: |x1-x2| + |y1-y2|
                        // Intensity difference: |I(x1,y1) - I(x2,y2)|
                        w = exp( -(abs(i+m-x) + abs(j+m-y)) / (2 * sigma_s * sigma_s)
                                 -abs(img.at<double>(i+m, j+m) - img.at<double>(x, y)) / (2 * sigma_r * sigma_r) );
                        weight += w;
                        arr += img.at<double>(x, y) * w;
                    }
                }
                dst.at<double>(i, j) = arr / weight;
            }
        }
        // Converts to int
        dst.convertTo(dst, CV_8U);
    }
    // 3 channels
    else if (channels == 3)
    {
        // Converts to float
        src.convertTo(img, CV_64FC3);
        dst = img.clone();
        // Extends borders by reflection
        int n = 2 * m + 1;
        copyMakeBorder(img, img, m, m, m, m, BORDER_REFLECT);
        double weight, w;
        Vec3d arr;
        for (int i = 0; i < dst.rows; i++)
        {
            for (int j = 0; j < dst.cols; j++)
            {
                arr = 0;
                weight = 0;
                for (int x = i; x < i + n; x++)
                {
                    for (int y = j; y < j + n; y++)
                    {
                        // Spatial difference: |x1-x2| + |y1-y2|
                        // Intensity difference: |B(x1,y1)-B(x2,y2)| + |G(x1,y1)-G(x2,y2)| + |R(x1,y1)-R(x2,y2)|
                        w = exp( -(abs(i+m-x) + abs(j+m-y)) / (2 * sigma_s * sigma_s)
                                 -(abs(img.at<Vec3d>(i+m, j+m)[0] - img.at<Vec3d>(x, y)[0])
                                 + abs(img.at<Vec3d>(i+m, j+m)[1] - img.at<Vec3d>(x, y)[1])
                                 + abs(img.at<Vec3d>(i+m, j+m)[2] - img.at<Vec3d>(x, y)[2]))
                                 / (2 * sigma_r * sigma_r) );
                        weight += w;
                        arr += img.at<Vec3d>(x, y) * w;
                    }
                }
                dst.at<Vec3d>(i, j) = arr / weight;
            }
        }
        // Converts to int
        dst.convertTo(dst, CV_8UC3);
    }
}

/// Fourier transformation utility:
/// Rearranges the outputs of dft by moving the zero-frequency component to the center of the array
// Source from http://www.cad.zju.edu.cn/home/gfzhang/course/computational-photography/lab2-filtering/filtering.html
void fftshift(const Mat &src, Mat &dst)
{
    dst.create(src.size(), src.type());
    int rows = src.rows, cols = src.cols;
    Rect roiTopBand, roiBottomBand, roiLeftBand, roiRightBand;
    if (rows % 2 == 0)
    {
        roiTopBand = Rect(0, 0, cols, rows / 2);
        roiBottomBand = Rect(0, rows / 2, cols, rows / 2);
    } else {
        roiTopBand = Rect(0, 0, cols, rows / 2 + 1);
        roiBottomBand = Rect(0, rows / 2 + 1, cols, rows / 2);
    }
    if (cols % 2 == 0)
    {
        roiLeftBand = Rect(0, 0, cols / 2, rows);
        roiRightBand = Rect(cols / 2, 0, cols / 2, rows);
    }  else {
        roiLeftBand = Rect(0, 0, cols / 2 + 1, rows);
        roiRightBand = Rect(cols / 2 + 1, 0, cols / 2, rows);
    }
    Mat srcTopBand = src(roiTopBand);
    Mat dstTopBand = dst(roiTopBand);
    Mat srcBottomBand = src(roiBottomBand);
    Mat dstBottomBand = dst(roiBottomBand);
    Mat srcLeftBand = src(roiLeftBand);
    Mat dstLeftBand = dst(roiLeftBand);
    Mat srcRightBand = src(roiRightBand);
    Mat dstRightBand = dst(roiRightBand);
    flip(srcTopBand, dstTopBand, 0);
    flip(srcBottomBand, dstBottomBand, 0);
    flip(dst, dst, 0);
    flip(srcLeftBand, dstLeftBand, 1);
    flip(srcRightBand, dstRightBand, 1);
    flip(dst, dst, 1);
}

/// Fourier transformation utility: visualization
void fftvisualize(const Mat &src, Mat &dst)
{
    Mat J(src.size(), CV_32FC2);
    // Fourier transformation
    dft(src, J, DFT_COMPLEX_OUTPUT);
    // Shifts zero-frequency component to center
    fftshift(J, J);
    Mat Mag;
    vector<Mat> K;
    // Separates real and virtual parts into K[0] and K[1]
    split(J, K);
    pow(K[0], 2, K[0]);
    pow(K[1], 2, K[1]);
    // Magnitude
    Mag = K[0]+K[1];
    // log_e (magnitude)
    log(Mag+1, dst);
    normalize(dst, dst, 1.0, 0.0, NORM_MINMAX);
}

int main()
{
    /// Test filters
    Mat img = imread("car.png");
    Mat ret1, ret2, ret3, ret4;

    BoxFilter(img, ret1, 3, 3);
    GaussianFilter(img, ret2, 3);
    MedianFilter(img, ret3, 3, 3);
    BilateralFilter(img, ret4, 3, 3);

//    imshow("original img", img);
//    waitKey(0);
//    imshow("box filter", ret1);
//    waitKey(0);
//    imshow("Gaussian filter", ret2);
//    waitKey(0);
//    imshow("median filter", ret3);
//    waitKey(0);
//    imshow("bilateral filter", ret4);
//    waitKey(0);

    Mat combine1, combine2, combine3;
    hconcat(ret1, ret2, combine1);
    hconcat(ret3, ret4, combine2);
    vconcat(combine1, combine2, combine3);

    imshow("result", combine3);
    imwrite("results.png", combine3);
    waitKey(0);


    /// Test Fourier transformation
    Mat I(512, 512, CV_32FC1);
    I = 0;
    Mat I1 = I.clone();
    Mat I2 = I.clone();
    Mat I3 = I.clone();
    // I: rectangle
    I(Rect(256-10, 256-30, 20, 60)) = 1.0;
    // I1: resized rectangle
    I1(Rect(256-20, 256-60, 40, 120)) = 1.0;
    // I2: rotated rectangle
    I2(Rect(256-30, 256-10, 60, 20)) = 1.0;
    // I3: replaced rectangle
    I3(Rect(256, 256, 20, 60)) = 1.0;

    Mat logMag, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, show;
    fftvisualize(I, logMag);
    vconcat(I, logMag, tmp1);
    fftvisualize(I1, logMag);
    vconcat(I1, logMag, tmp2);
    fftvisualize(I2, logMag);
    vconcat(I2, logMag, tmp3);
    fftvisualize(I3, logMag);
    vconcat(I3, logMag, tmp4);

    hconcat(tmp1, tmp2, tmp5);
    hconcat(tmp3, tmp4, tmp6);
    hconcat(tmp5, tmp6, show);

    show.convertTo(show, CV_8U, 255, 0);
    imshow("logMag", show);
    imwrite("logMag.png", show);
    waitKey(0);

    return 0;
}
