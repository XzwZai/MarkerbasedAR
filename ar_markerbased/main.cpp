#include <opencv2/opencv.hpp>
#include <iostream>
#include "MarkerDetector.h"
using namespace cv;
using namespace std;




int main()
{
	Mat img = imread("D:/WorkSpace/data/markerbased_ar/3.jpg");
	MarkerDetector md;
	vector<Marker> markers;
	md.findMarkers(img, markers);

	waitKey(0);
}