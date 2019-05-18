#ifndef GNMETHOD_RESIDUALFUNCTIONxxxx_H
#define GNMETHOD_RESIDUALFUNCTIONxxxx_H

#include "hw3_gn.h"

#define TEST_DATA_NUM 753
#define VARIABLE_NUM 3

using namespace std;

// x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1
class ResidualFunctionxxxx: public ResidualFunction {
public:
    int nR() const override;
    int nX() const override;
    void eval(double *R, double *J, double *X) override;
    ResidualFunctionxxxx();

private:
    double testData[TEST_DATA_NUM][VARIABLE_NUM];
};

#endif //GNMETHOD_RESIDUALFUNCTIONxxxx_H
