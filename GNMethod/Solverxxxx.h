#ifndef GNMETHOD_SOLVERxxxx_H
#define GNMETHOD_SOLVERxxxx_H

#include "hw3_gn.h"

class Solverxxxx: public GaussNewtonSolver {
public:
    double solve(
            ResidualFunction *f,
            double *X,
            GaussNewtonParams param,
            GaussNewtonReport *report
            ) override;
};

#endif //GNMETHOD_SOLVERxxxx_H
