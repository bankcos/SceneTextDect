#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <stdio.h>
#include <stdlib.h>
#include "region.cpp"
#include "distance.cpp"

#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int showImg(Mat img)
{

	namedWindow("Display", WINDOW_AUTOSIZE);
	//	cvShowImage("Display", transimg1);

	resize(img, img, Size(640, 480), INTER_CUBIC);
	imshow("Display", img);
	waitKey(0);
	return 0;
}


Mat  tSIFT(String path)
{
	Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
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
	Mat img1;
	drawKeypoints(img, tSIFTkp, img1, Scalar::all(-1), 4);
	//FeaturesExtract
	SiftDescriptorExtractor extractor;
	//提取特征向量
	extractor.compute(img,tSIFTkp,des);

	showImg(img1);

	return des;
}

int main(){

	string samplePath = "d:\\project\\traindata\\";
	string path;
	char sampleName[256];
	Mat a;
	for (int i = 100; i < 103; i++)
	{
		memset(sampleName, '\0', sizeof(char)* 256);
		sprintf_s(sampleName, "%d.jpg", i);
		//文件路径
		path = samplePath + sampleName;
		
		auto c=tSIFT(path);
		a.push_back(c);
		cout << i << endl;
	}
	
	return 0;
}

