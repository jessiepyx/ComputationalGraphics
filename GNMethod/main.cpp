/*
 * Course assigment website:
 * http://www.cad.zju.edu.cn/home/gfzhang/course/computational-photography/lab5-gauss-newton/gauss-newton.html
 */

#include <iostream>
#include "opencv2/opencv.hpp"
#include "Solverxxxx.h"
#include "ResidualFunctionxxxx.h"

using namespace std;
using namespace cv;

/* linear least square solution, for verification */
void verify(double *X)
{
    // read data
    double testData[TEST_DATA_NUM][VARIABLE_NUM];
    fstream file("ellipse753.txt", ios::in);
    if (!file.is_open())
    {
        cout << "Error opening file" << endl;
        exit(1);
    }
    for (auto &i : testData)
    {
        for (double &j : i)
        {
            if (!(file >> j))
            {
                cout << "Error reading file" << endl;
                exit(1);
            }
            j *= j; // squared
        }
    }

    // solve linear equations for X = (1/a^2, 1/b^2, 1/c^2)
    Mat_<double> mX(VARIABLE_NUM, 1, X);
    Mat_<double> mA(TEST_DATA_NUM, VARIABLE_NUM, (double*)testData);
    Mat_<double> mb = Mat_<double>::ones(TEST_DATA_NUM, 1);
    if (!cv::solve(mA, mb, mX, CV_SVD)) // AX = b
    {
        cout << "Error solving AX=b" << endl;
        exit(1);
    }
}

int main()
{
    double X[3] = {4, 4, 1};
    ResidualFunctionxxxx myfunc;
    Solverxxxx mysolver;
    GaussNewtonParams param;
//    param.verbose = true;
    GaussNewtonReport report;

    double res = mysolver.solve(&myfunc, X, param, &report);

    cout << "res = " << res << endl;
    cout << "X = (" << X[0] << ", " << X[1] << ", " << X[2] << ")" << endl;
    cout << report.n_iter << " iterations" << endl;
    cout << "stop type = " << report.stop_type << endl;

    double Y[3] = {4, 4, 1};
    verify(Y);
    cout << "(verify)1/X = (" << Y[0] << ", " << Y[1] << ", " << Y[2] << ")" << endl;
    cout << "(verify)X = (" << 1/Y[0] << ", " << 1/Y[1] << ", " << 1/Y[2] << ")" << endl;
    return 0;
}