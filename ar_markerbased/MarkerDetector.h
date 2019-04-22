#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Marker.h"
#include "Tools.h"
using namespace cv;
using namespace std;

class MarkerDetector
{
public:
	float m_minContourLengthAllowed;	
	Size markerSize;

	Mat sourceImage;
	Mat m_grayscaleImage;
	Mat m_thresholdImg;
	Mat canonicalMarkerImage;

	vector<vector<Point>> contours;
	vector<Point3f> m_markerCorners3d;
	vector<Point2f> m_markerCorners2d;
	Mat camMatrix;
	Mat distCoeff;

	MarkerDetector() : markerSize(100, 100)
	{
		m_markerCorners2d.push_back(cv::Point2f(0, 0));
		m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, 0));
		m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, markerSize.height - 1));
		m_markerCorners2d.push_back(cv::Point2f(0, markerSize.height - 1));

		m_markerCorners3d.push_back(cv::Point3f(-0.5f, -0.5f, 0));
		m_markerCorners3d.push_back(cv::Point3f(+0.5f, -0.5f, 0));
		m_markerCorners3d.push_back(cv::Point3f(+0.5f, +0.5f, 0));
		m_markerCorners3d.push_back(cv::Point3f(-0.5f, +0.5f, 0));

		((Mat)Mat::zeros(4, 1, CV_32F)).copyTo(distCoeff);

		float m_intrinsic[4][4];
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				m_intrinsic[i][j] = 0;

		m_intrinsic[0][0] = 6.24860291e+02 * (640. / 352.);
		m_intrinsic[1][1] = 6.24860291e+02 * (480. / 288.);
		m_intrinsic[0][2] = 640 * 0.5f;
		m_intrinsic[1][2] = 480 * 0.5f;
		Mat(3, 3, CV_32F, const_cast<float*>(&m_intrinsic[0][0])).copyTo(camMatrix);
	}

	bool findMarkers(Mat& sourceImg, vector<Marker>& detectedMarkers)
	{
		sourceImg.copyTo(sourceImage);
		cvtColor(sourceImg, m_grayscaleImage, CV_RGB2GRAY);
		threshold(m_grayscaleImage, m_thresholdImg, 0, 255, THRESH_BINARY_INV | THRESH_TRIANGLE);
		myFindContours(m_thresholdImg, contours, 20);
		findCandidates(contours, detectedMarkers);
		recognizeMarkers(m_grayscaleImage, detectedMarkers);
		estimatePosition(detectedMarkers);
		for (int i = 0; i < detectedMarkers.size(); i++)
		{
			detectedMarkers[i].draw(sourceImage);
			imshow("detected", sourceImage);
			waitKey(0);
		}
		return true;
	}

	void myFindContours(const Mat& thresholdImg, vector<vector<Point>>& contours, int minPointsAllowed)
	{
		vector<vector<Point>> allContours;
		findContours(thresholdImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		contours.clear();
		for (int i = 0; i < allContours.size(); i++)
		{
			if (allContours[i].size() > minPointsAllowed)
			{
				contours.push_back(allContours[i]);
			}
		}
		/*Mat testImg = Mat::zeros(thresholdImg.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++) {
			drawContours(testImg, contours, i, Scalar(255, 255, 255), -1, 8);
			imshow("test", testImg);
			waitKey(0);
		}*/
		Mat dstImg = Mat::zeros(thresholdImg.size(), CV_8UC3);
		drawContours(dstImg, contours, -1, Scalar(255, 255, 255), 1, 8);
		imshow("contours", dstImg);
	}

	void findCandidates
	(
		const vector<vector<Point>>& contours,
		std::vector<Marker>& detectedMarkers
	)
	{
		std::vector<cv::Point>  approxCurve;
		std::vector<Marker>     possibleMarkers;

		// For each contour, analyze if it is a parallelepiped likely to be the marker
		for (size_t i = 0; i < contours.size(); i++)
		{
			// Approximate to a polygon
			double eps = contours[i].size() * 0.05;
			cv::approxPolyDP(contours[i], approxCurve, eps, true);

			// We interested only in polygons that contains only four points
			if (approxCurve.size() != 4)
				continue;

			// And they have to be convex
			if (!cv::isContourConvex(approxCurve))
				continue;

			// Ensure that the distance between consecutive points is large enough
			float minDist = std::numeric_limits<float>::max();

			for (int i = 0; i < 4; i++)
			{
				cv::Point side = approxCurve[i] - approxCurve[(i + 1) % 4];
				float squaredSideLength = side.dot(side);
				minDist = std::min(minDist, squaredSideLength);
			}

			// Check that distance is not very small
			if (minDist < 100)
				continue;

			// All tests are passed. Save marker candidate:
			Marker m;

			for (int i = 0; i < 4; i++)
				m.points.push_back(cv::Point2f(approxCurve[i].x, approxCurve[i].y));

			// Sort the points in anti-clockwise order
			// Trace a line between the first and second point.
			// If the third point is at the right side, then the points are anti-clockwise
			cv::Point v1 = m.points[1] - m.points[0];
			cv::Point v2 = m.points[2] - m.points[0];

			double o = (v1.x * v2.y) - (v1.y * v2.x);

			if (o < 0.0)		 //if the third point is in the left side, then sort in anti-clockwise order
				std::swap(m.points[1], m.points[3]);

			possibleMarkers.push_back(m);
		}



		// Remove these elements which corners are too close to each other.  
		// First detect candidates for removal:
		std::vector< std::pair<int, int> > tooNearCandidates;
		for (size_t i = 0; i < possibleMarkers.size(); i++)
		{
			const Marker& m1 = possibleMarkers[i];

			//calculate the average distance of each corner to the nearest corner of the other marker candidate
			for (size_t j = i + 1; j < possibleMarkers.size(); j++)
			{
				const Marker& m2 = possibleMarkers[j];

				float distSquared = 0;

				for (int c = 0; c < 4; c++)
				{
					cv::Point v = m1.points[c] - m2.points[c];
					distSquared += v.dot(v);
				}

				distSquared /= 4;

				if (distSquared < 100)
				{
					tooNearCandidates.push_back(std::pair<int, int>(i, j));
				}
			}
		}

		// Mark for removal the element of the pair with smaller perimeter
		std::vector<bool> removalMask(possibleMarkers.size(), false);

		for (size_t i = 0; i < tooNearCandidates.size(); i++)
		{
			float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
			float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);

			size_t removalIndex;
			if (p1 > p2)
				removalIndex = tooNearCandidates[i].second;
			else
				removalIndex = tooNearCandidates[i].first;

			removalMask[removalIndex] = true;
		}

		// Return candidates
		detectedMarkers.clear();
		for (size_t i = 0; i < possibleMarkers.size(); i++)
		{
			if (!removalMask[i])
				detectedMarkers.push_back(possibleMarkers[i]);
		}
		/*Mat dstImg = Mat::zeros(sourceImage.size(), CV_8UC3);
		for (int i = 0; i < detectedMarkers.size(); i++)
		{
			detectedMarkers[i].draw(dstImg);
			imshow("possible", dstImg);
			waitKey(0);
		}*/
	}

	void recognizeMarkers(const cv::Mat& grayscale, std::vector<Marker>& detectedMarkers)
	{
		std::vector<Marker> goodMarkers;
		for (size_t i = 0; i < detectedMarkers.size(); i++)
		{
			Marker& marker = detectedMarkers[i];

			// Find the perspective transformation that brings current marker to rectangular form
			cv::Mat markerTransform = cv::getPerspectiveTransform(marker.points, m_markerCorners2d);

			// Transform image to get a canonical marker image
			cv::warpPerspective(grayscale, canonicalMarkerImage, markerTransform, markerSize);
			int nRotations;
			int id = Marker::getMarkerId(canonicalMarkerImage, nRotations);
			if (id != -1)
			{
				marker.id = id;
				//sort the points so that they are always in the same order no matter the camera orientation
				std::rotate(marker.points.begin(), marker.points.begin() + 4 - nRotations, marker.points.end());

				goodMarkers.push_back(marker);
			}
			/*imshow("warp", canonicalMarkerImage);
			waitKey(0);*/
		}

		if (goodMarkers.size() > 0)
		{
			std::vector<cv::Point2f> preciseCorners(4 * goodMarkers.size());

			for (size_t i = 0; i < goodMarkers.size(); i++)
			{
				const Marker& marker = goodMarkers[i];

				for (int c = 0; c < 4; c++)
				{
					preciseCorners[i * 4 + c] = marker.points[c];
				}
			}

			cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01);
			cv::cornerSubPix(grayscale, preciseCorners, cvSize(5, 5), cvSize(-1, -1), termCriteria);

			// Copy refined corners position back to markers
			for (size_t i = 0; i < goodMarkers.size(); i++)
			{
				Marker& marker = goodMarkers[i];

				for (int c = 0; c < 4; c++)
				{
					marker.points[c] = preciseCorners[i * 4 + c];
				}
			}
		}
		detectedMarkers = goodMarkers;
	}



	void estimatePosition(std::vector<Marker>& detectedMarkers)
	{
		
		for (size_t i = 0; i < detectedMarkers.size(); i++)
		{
			Marker& m = detectedMarkers[i];

			cv::Mat Rvec;
			cv::Mat_<float> Tvec;
			cv::Mat raux, taux;
			cv::solvePnP(m_markerCorners3d, m.points, camMatrix, distCoeff, raux, taux);
			raux.convertTo(Rvec, CV_32F);
			taux.convertTo(Tvec, CV_32F);

			cv::Mat_<float> rotMat(3, 3);
			cv::Rodrigues(Rvec, rotMat);

			// Copy to transformation matrix
			for (int col = 0; col < 3; col++)
			{
				for (int row = 0; row < 3; row++)
				{
					m.transformation.r().mat[row][col] = rotMat(row, col); // Copy rotation component
				}
				m.transformation.t().data[col] = Tvec(col); // Copy translation component
			}

			// Since solvePnP finds camera location, w.r.t to marker pose, to get marker pose w.r.t to the camera we invert it.
			m.transformation = m.transformation.getInverted();
		}
	}
};

