/*
 * @file main.cpp
 * @brief panorama image mosaics
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

int main()
{
    for (int i : {1, 2})
    {
        cout << "Starting to process Case " << i << "..." << endl;

        // path
        String dir = "./panorama-data" + to_string(i) + "/";
        String img_path = dir + "*.JPG";

        // get all filenames
        vector<String> img_names;
        glob(img_path, img_names);

        // read images
        vector<Mat> img_vec;
        for (const auto& img : img_names)
        {
            img_vec.emplace_back(imread(img));
        }
        cout << "Finish reading " << img_vec.size() << " images." << endl;

//        for (const auto& img : img_vec)
//        {
//            cout << img.size() << endl;
//        }

        // read focal length
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

        Mat img_out;

        // make panorama
        Panoramaxxxx pm;
        pm.makePanorama(img_vec, img_out, f);
        cout << "Finished processing Case " << i << "." << endl;

        // show result
        imshow("panorama " + to_string(1), img_out);
        waitKey(0);
    }
    return 0;
}
