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
/*	
	string samplePath = "d:\\project\\MSRG\\MSRG_";
	string path;
	char sampleName[256];
	for (int i = 1; i < 200; i++)
	{
		memset(sampleName, '\0', sizeof(char)* 256);
		sprintf(sampleName, "%04i.jpg", i);
		path = samplePath + sampleName;
		cout << path << endl;
	}
*/
	FileStorage fs("d:\\project\\test.xml", FileStorage::WRITE);
	fs << "frameCount" << 5;

	Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1); //又一种Mat初始化方式
	Mat distCoeffs = (Mat_<double>(5, 1) << 0.1, 0.01, -0.001, 0, 0);
	fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;

	//features为一个大小为3的向量,其中每个元素由随机数x,y和大小为8的uchar数组组成
	fs << "features" << "[";
	for (int i = 0; i < 3; i++)
	{
		int x = rand() % 640;
		int y = rand() % 480;
		uchar lbp = rand() % 256;
		fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";
		for (int j = 0; j < 8; j++)
			fs << ((lbp >> j) & 1);
		fs << "]" << "}";
	}
	fs << "]";
	fs.release();


	system("pause");
}