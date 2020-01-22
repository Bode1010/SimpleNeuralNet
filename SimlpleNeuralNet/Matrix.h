#include <iostream>
#include <vector>
#include <cstdlib>//for rand function
#pragma once
using namespace std;

class Matrix
{
public:
	Matrix() {};
	Matrix(int, int, bool);//rows, columns, random(true or false)
	Matrix operator+(const Matrix &);
	Matrix operator*(const Matrix &);
	Matrix operator-(const Matrix &);
	Matrix operator/(float);
	Matrix operator*(float);
	void operator=(const Matrix &);
	void clear();
	void print();
	float AddCol(int col);
	void randomize();
	vector<vector<float>> vals;
	~Matrix();

	Matrix EWM(const Matrix&);//Element wise multiplication
};

