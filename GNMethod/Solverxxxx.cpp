#include <iostream>
#include "Solverxxxx.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

double Solverxxxx::solve(ResidualFunction *f, double *X, GaussNewtonParams param, GaussNewtonReport *report)
{
    int nX = f->nX();
    int nR = f->nR();
    int nJ = nR * nX;
    double R[nR];
    double J[nJ];
    Mat_<double> deltaX(nX, 1);
    Mat_<double> mJ(nR, nX, J);
    Mat_<double> mR(nR, 1, R);

    int n = 0;
    double step = 0;
    while (n < param.max_iter)
    {
        n++;

        // calculate R, J
        f->eval(R, J, X);

        // solve deltaX
        Mat_<double> mJT(mJ.t());
        if (!cv::solve(mJT * mJ, mJT * (-mR), deltaX, CV_SVD)) // JT * J * deltaX = JT * (-R)
        {
            if (report != nullptr) // ERROR solving deltaX, return
            {
                report->n_iter = n;
                report->stop_type = report->STOP_NUMERIC_FAILURE;
            }
            return norm(mR, NORM_L2);
        }
        if (param.verbose)
        {
            cout << "iteration #" << n << endl;
            cout << "deltaX = " << deltaX << endl;
        }

        // decide if stop iteration
        if (norm(mR, NORM_INF) <= param.residual_tolerance)
        {
            if (report != nullptr) // reach residual tolerance, return
            {
                report->n_iter = n;
                report->stop_type = report->STOP_RESIDUAL_TOL;
            }
            return norm(mR, NORM_L2);
        }
        if (norm(deltaX, NORM_INF) <= param.gradient_tolerance)
        {
            if (report != nullptr) // reach grad tolerance, return
            {
                report->n_iter = n;
                report->stop_type = report->STOP_GRAD_TOL;
            }
            return norm(mR, NORM_L2);
        }

        // line search for optimal step length (not implemented yet)
        if (param.exact_line_search)
        {
            step = 1;
        }
        else
        {
            step = 0.2;
        }

        // update X
        for (int i = 0; i < nX; i++)
        {
            X[i] += step * deltaX(i, 0);
        }
        if (param.verbose)
        {
            cout << "X = ";
            for (int i = 0; i < nX; i++)
            {
                cout << X[i] << " ";
            }
            cout << endl;
        }
    }

    if (report != nullptr)
    {
        report->n_iter = n;
        report->stop_type = report->STOP_NO_CONVERGE;
    }
    return norm(mR, NORM_L2);
}
