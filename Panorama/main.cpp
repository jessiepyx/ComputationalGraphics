/**
 * @file main.cpp
 * @brief panorama image stitching
 * @author Jessie Peng
 */

#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "my_pa.h"

#define DEFAULT_FOCAL_LENGTH 512.89

using namespace std;
using namespace cv;

#define TEST_NO 2

int main()
{
    cout << "Starting to process Case " << TEST_NO << "..." << endl;

    /// path
    String dir = "./panorama-data" + to_string(TEST_NO) + "/";
    String img_path = dir + "*.JPG";

    /// get all filenames
    vector<String> img_names;
    glob(img_path, img_names);

    /// read images
    vector<Mat> img_vec;
    for (const auto& img : img_names)
    {
        img_vec.push_back(imread(img));
    }
    int n = img_vec.size();
    cout << "Finish reading " << n << " images." << endl;

    /// read focal length
    double f;
    fstream file(dir + "K.txt", ios::in);
    if (file.is_open())
    {
        if (file >> f)
        {
            cout << "Focal length: " << f << endl;
        }
        else
        {
            f = DEFAULT_FOCAL_LENGTH;
            cout << "Warning: Failed to read K.txt,"
                    "setting focal length to " << f << " by default." << endl;

        }
    }
    else
    {
        f = DEFAULT_FOCAL_LENGTH;
        cout << "Warning: Failed to open K.txt,"
                "setting focal length to " << f << " by default." << endl;
    }

    // estimate the size of stitched image
    Mat img_out = Mat::zeros(img_vec[0].rows * (1 + n * 0.05), img_vec[0].cols * (1 + n * 0.15), CV_8UC3);

    /// make panorama
    Panoramaxxxx pm;
    pm.makePanorama(img_vec, img_out, f);
    cout << "Finished processing Case " << TEST_NO << "." << endl;

    /// show result
    imshow("panorama " + to_string(TEST_NO), img_out);
    waitKey(0);
    destroyAllWindows();
    imwrite("./panorama_" + to_string(TEST_NO) + ".png", img_out);

    return 0;
}
