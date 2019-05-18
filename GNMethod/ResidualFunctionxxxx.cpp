#include <iostream>
#include <fstream>
#include "ResidualFunctionxxxx.h"

using namespace std;

int ResidualFunctionxxxx::nR() const
{
    return TEST_DATA_NUM;
}

int ResidualFunctionxxxx::nX() const
{
    return VARIABLE_NUM; // a^2, b^2, c^2
}

void ResidualFunctionxxxx::eval(double *R, double *J, double *X)
{
    // x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1
    for (int i = 0; i < TEST_DATA_NUM; i++)
    {
        R[i] = -1;
        for (int j = 0; j < VARIABLE_NUM; j++)
        {
            R[i] += testData[i][j] * testData[i][j] / X[j];
        }
    }
    for (int i = 0; i < TEST_DATA_NUM; i++)
    {
        for (int j = 0; j < VARIABLE_NUM; j++)
        {
            J[i*VARIABLE_NUM+j] = (-1) * testData[i][j] * testData[i][j] / X[j] / X[j];
        }
    }
}

ResidualFunctionxxxx::ResidualFunctionxxxx()
{
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
        }
    }
}
