/*
 * Test G-S Method with
 *
 *      10  -1   2   0
 * A =  -1  11  -1   3
 *       2  -1  10  -1
 *       0   3  -1   8
 *
 * b = [ 1, 2, -1, 1 ]
 *
 * x = [ 1, 2, -1, 1]
 */

#include <iostream>
#include "GaussSeidel.h"

using namespace std;

int main()
{
    MySparseMatrix<double> A(4, 4);
    A.insert(10, 1, 1);
    A.insert(-1, 1, 2);
    A.insert(2, 1, 3);
    A.insert(-1, 2, 1);
    A.insert(11, 2, 2);
    A.insert(-1, 2, 3);
    A.insert(3, 2, 4);
    A.insert(2, 3, 1);
    A.insert(-1, 3, 2);
    A.insert(10, 3, 3);
    A.insert(-1, 3, 4);
    A.insert(3, 4, 2);
    A.insert(-1, 4, 3);
    A.insert(8, 4, 4);
    A.printInfo();

    vector<double> b;
    b.push_back(6);
    b.push_back(25);
    b.push_back(-11);
    b.push_back(15);

    GaussSeidel<double> solver;
    if (solver.solve(A, b, 5, true))
    {
        vector<double> x = solver.getResult();
        for (auto i : x) {
            cout << i << " ";
        }
        cout << endl;
        cout << "iteration number: " << solver.getIters() << endl;
    }
    if (solver.solve(A, b, 5, true))
    {
        vector<double> x = solver.getResult();
        for (auto i : x) {
            cout << i << " ";
        }
        cout << endl;
        cout << "iteration number: " << solver.getIters() << endl;
    }

    return 0;
}