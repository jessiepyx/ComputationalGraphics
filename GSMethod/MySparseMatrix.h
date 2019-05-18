/*
 * simple sparce matrix implementation
 * using Compressed Row Storage (CRS)
 * row_ptr: row pointers, row_ptr[0] is ignored, row_ptr[row_cnt+1] is to mark the end
 * col_val: column-value pairs, containing (0, 0) pairs to mark the dummy head of each row
 */

#ifndef GSMETHOD_MYSPARSEMATRIX_H
#define GSMETHOD_MYSPARSEMATRIX_H

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

template <typename T> class MySparseMatrix {
public:
    explicit MySparseMatrix(int m = 10, int n = 10);
    T at(int row, int col) const;
    void insert(T val, int row, int col);
    bool initializeFromVector(const vector<int> &rows, const vector<int> &cols, const vector<T> &vals);
    void printInfo() const;
    int getRowsCnt() const { return rows_cnt; }
    int getColsCnt() const { return cols_cnt; }
private:
    vector<pair<int, T>> col_val;
    vector<int> row_ptr;
    int rows_cnt;
    int cols_cnt;
    auto getIterator(int ind) const;
};

template <typename T> MySparseMatrix<T>::MySparseMatrix(int m, int n): rows_cnt(m), cols_cnt(n)
{
    row_ptr.push_back(0);
    for (int i = 0; i <= rows_cnt; i++)
    {
        row_ptr.push_back(i);
        col_val.push_back(make_pair(0, 0));
    }
}

template <typename T> T MySparseMatrix<T>::at(int row, int col) const
{
    int start = row_ptr[row];
    int end = row_ptr[row+1];
    for (int i = start; i < end; i++)
    {
        if (col_val[i].first == col)
        {
            return col_val[i].second;
        }
    }
    return 0;
}

template <typename T> void MySparseMatrix<T>::insert(T val, int row, int col)
{
    int start = row_ptr[row];
    int end = row_ptr[row+1];
    int i;
    for (i = start; i < end; i++)
    {
        if (col_val[i].first == col)
        {
            if (val == 0) // 1. non-zero -> zero
            {
                auto j = getIterator(i);
                col_val.erase(j); // remove pair (col, val)
                for (int k = row+1; k <= rows_cnt+1; k++)
                {
                    row_ptr[k]--; // update row pointers
                }
            }
            else // 2. non-zero -> non-zero
            {
                col_val[i].second = val; // update val
            }
            return;
        }
        if (col_val[i].first > col)
        {
            break;
        }
    }
    if (val != 0) // 3. zero -> non-zero
    {
        auto j = getIterator(i);
        col_val.insert(j, make_pair(col, val)); // add pair (col, val)
        for (int k = row+1; k <= rows_cnt+1; k++)
        {
            row_ptr[k]++; // update row pointers
        }
    }
}

template <typename T> bool MySparseMatrix<T>::initializeFromVector(
        const vector<int> &rows, const vector<int> &cols, const vector<T> &vals)
{
    unsigned long len = rows.size();
    if (cols.size() != len || vals.size() != len)
    {
        return false;
    }
    for (int i = 0; i < len; i++)
    {
        insert(vals[i], rows[i], cols[i]);
    }
    return true;
}

template <typename T> void MySparseMatrix<T>::printInfo() const
{
    cout << "row_ptr: ";
    for (auto i : row_ptr)
    {
        cout << i << " ";
    }
    printf("\ncol: ");
    for (auto i : col_val)
    {
        printf("%6d ", i.first);
    }
    printf("\nval: ");
    for (auto i : col_val)
    {
        printf("%6.2f ", (double)i.second);
    }
    printf("\n");
}

template <typename T> auto MySparseMatrix<T>::getIterator(int ind) const
{
    auto j = col_val.begin();
    for (int k = 0; k < ind; k++)
    {
        j++;
    }
    return j;
}

#endif //GSMETHOD_MYSPARSEMATRIX_H



















