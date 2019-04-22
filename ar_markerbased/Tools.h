#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

float perimeter(const std::vector<cv::Point2f> &a)
{
	float sum = 0, dx, dy;

	for (size_t i = 0; i < a.size(); i++)
	{
		size_t i2 = (i + 1) % a.size();

		dx = a[i].x - a[i2].x;
		dy = a[i].y - a[i2].y;

		sum += sqrt(dx*dx + dy * dy);
	}

	return sum;
}

bool isInto(cv::Mat &contour, std::vector<cv::Point2f> &b)
{
	for (size_t i = 0; i < b.size(); i++)
	{
		if (cv::pointPolygonTest(contour, b[i], false) > 0) return true;
	}
	return false;
}

struct Matrix44
{
	union
	{
		float data[16];
		float mat[4][4];
	};

	Matrix44 getTransposed() const
	{
		Matrix44 t;

		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				t.mat[i][j] = mat[j][i];

		return t;
	}

	static Matrix44 identity()
	{
		Matrix44 eye;

		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				eye.mat[i][j] = i == j ? 1 : 0;

		return eye;
	}

	Matrix44 getInvertedRT() const
	{
		Matrix44 t = identity();

		for (int col = 0; col < 3; col++)
		{
			for (int row = 0; row < 3; row++)
			{
				// Transpose rotation component (inversion)
				t.mat[row][col] = mat[col][row];
			}

			// Inverse translation component
			t.mat[3][col] = -mat[3][col];
		}
		return t;
	}
};

struct Matrix33
{
	union
	{
		float data[9];
		float mat[3][3];
	};
	
	static Matrix33 identity()
	{
		Matrix33 eye;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				eye.mat[i][j] = i == j ? 1 : 0;
			}
		}
		return eye;
	}
	Matrix33 getTransposed() const
	{
		Matrix33 t;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				t.mat[i][j] = mat[j][i];
		return t;
	}
};

struct Vector4
{
	float data[4];
};

struct Vector3
{
	float data[3];

	static Vector3 zero()
	{
		Vector3 v;
		for (int i = 0; i < 3; i++)
		{
			v.data[i] = 0;
		}
	}
	Vector3 operator-() const
	{
		Vector3 v;
		for (int i = 0; i < 3; i++)
		{
			v.data[i] = -data[i];
		}
	}
};

class Transformation
{
public:
	Transformation() {}
	Transformation(Matrix33& r, Vector3& t) : m_rotation(r), m_translation(t) {}
	Matrix33& r() { return m_rotation; }
	Vector3& t() { return m_translation; }
	Matrix44 getMat44() const
	{
		Matrix44 res = Matrix44::identity();

		for (int col = 0; col < 3; col++)
		{
			for (int row = 0; row < 3; row++)
			{
				// Copy rotation component
				res.mat[row][col] = m_rotation.mat[row][col];
			}

			// Copy translation component
			res.mat[3][col] = m_translation.data[col];
		}

		return res;
	}

	Transformation getInverted() const
	{
		Vector3 t = -m_translation;
		Matrix33 r = m_rotation.getTransposed();
		return Transformation(r, t);		
	}
private:
	Matrix33 m_rotation;
	Vector3 m_translation;
};