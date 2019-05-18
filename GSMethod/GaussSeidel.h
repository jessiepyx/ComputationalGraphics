/*
 * Gauss-Seidel Method (applies to class MySparseMatrix)
 * x_i^{k+1} = (1/a_{ii}) *
 *             ( b_i - Sigma_{j<i}(a_{ij}*x_j^{k+1}) - Sigma_{j>i}(a_{ij}*x_j^k) )
 */

#ifndef GSMETHOD_GAUSSSEIDEL_H
#define GSMETHOD_GAUSSSEIDEL_H

#include <iostream>
#include "MySparseMatrix.h"

using namespace std;

template <typename T> class GaussSeidel {
public:
    explicit GaussSeidel(int len = 4);
    explicit GaussSeidel(const vector<T> &initval);
    bool solve(const MySparseMatrix<T> &A, const vector<T> &b, int iter = 20, bool verbose = false);
    vector<T> getResult() const { return x; }
    int getIters() const { return iters; }
private:
    int n;
    vector<T> x;
    int iters;
    bool zeroInDiag(const MySparseMatrix<T> &A) const;
};

template <typename T> GaussSeidel<T>::GaussSeidel(int len): n(len), iters(0)
{
    for (int i = 0; i < n; i++)
    {
        x.push_back(1); // initial values are 1 by default
    }
}

template <typename T> GaussSeidel<T>::GaussSeidel(const vector<T> &initval): n((int)initval.size()), iters(0)
{
    for (int i = 0; i < n; i++)
    {
        x.push_back(initval[i]);
    }
}

template <typename T> bool GaussSeidel<T>::solve(const MySparseMatrix<T> &A, const vector<T> &b, int iter, bool verbose)
{
    if (A.getColsCnt() != n || A.getRowsCnt() != n || b.size() != n || zeroInDiag(A))
    {
        return false;
    }
    int cnt;
    bool stop = false;
    for (cnt = 0; cnt < iter; cnt++)
    {
        if (verbose)
        {
            for (auto i : x)
            {
                cout << i << " ";
            }
            cout << endl;
        }
        if (stop)
        {
            break;
        }
        stop = true;
        double sigma;
        double tmp;
        for (int i = 1; i <= n; i++)
        {
            sigma = 0;
            for (int j = 1; j <= n; j++)
            {
                if (j != i)
                {
                    sigma += A.at(i, j) * x[j-1];
                }
            }
            tmp = (double)(b[i-1] - sigma) / A.at(i, i);
            if (tmp != x[i-1])
            {
                stop = false;
                x[i-1] = tmp;
            }
        }
    }
    iters += cnt;
    return true;
}

template <typename T> bool GaussSeidel<T>::zeroInDiag(const MySparseMatrix<T> &A) const
{
    for (int i = 1; i <= A.getRowsCnt(); i++)
    {
        if (A.at(i, i) == 0) // if there's a zero in A's diagonal
        {
            return true;
        }
    }
    return false;
}

#endif //GSMETHOD_GAUSSSEIDEL_H
