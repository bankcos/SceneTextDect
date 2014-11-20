#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace cv;
using namespace std;

void main()
{
	Mat img1;
	Mat img = imread("d:\\project\\gray.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//特征点描述符
	Mat des;
	if (!img.data){
		std::cout << "Can't open" << std::endl;
		system("Pause");
		exit(0);
	}

	SiftFeatureDetector detector;
	std::vector<KeyPoint> tSIFTkp;
	detector.detect(img, tSIFTkp);

	drawKeypoints(img, tSIFTkp, img1, Scalar::all(-1), 4);
	//FeaturesExtract
	SiftDescriptorExtractor extractor;
	//提取特征向量
	extractor.compute(img, tSIFTkp, des);
	Mat *a = &des;

	FileStorage fs("d:\\project\\test.xml", FileStorage::WRITE);
	fs << "sss" << img;
	fs.release();

	FileStorage fs("d:\\project\\test.xml", FileStorage::READ);
	fs["sss"] >>img1;
	fs.release();

	namedWindow("aa", WINDOW_AUTOSIZE);
	imshow("aa",img1);
	waitKey(0);
}