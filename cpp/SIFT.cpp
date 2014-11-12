#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <stdio.h>
#include <stdlib.h>
#include "region.cpp"
#include "distance.cpp"

using namespace cv;
using namespace std;

int numOfRegion;//聚类数量

float dis2Points(Point a, Point b){
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}
vector<KeyPoint> regionGrowing(Mat img, vector<KeyPoint> p)
{
	vector<KeyPoint> kp3;
	float minGrowDis =20;
	int minGrayDis = 10;
	//种子点与目标点的灰度值
	int seedGray, tmpGray;
	//种子点
	KeyPoint seed;
	//k分类区域

	//候选点非空
	while (!p.empty())
	{
		seed = p[p.size() - 1];
		seed.class_id = numOfRegion;
		p.pop_back();
		//$$$$$$$$$$
		seedGray = img.at<uchar>(seed.pt.y, seed.pt.x);
		//将种子点放入新容器
		kp3.push_back(seed);
		//采用遍历算法
		for (int i = 0; i < p.size(); i++)
		{
			float dis;
			KeyPoint tmpPoint;
			tmpPoint = p[i];
			tmpGray = img.at<uchar>(tmpPoint.pt.y, tmpPoint.pt.x);
			//计算2两点间距离
			dis = dis2Points(seed.pt, tmpPoint.pt);
			//通过距离和灰度差判断是否是我族类
			//
			if (dis < minGrowDis && abs(seedGray - tmpGray) < minGrayDis)
			{
				tmpPoint.class_id = numOfRegion;
				kp3.push_back(tmpPoint);
				p.erase(p.begin() + i);
				i--;
			}

		}

		//聚类种类++
		numOfRegion++;
	}
	return kp3;
}

//求局部熵
float entropy(Mat Roi, KeyPoint p)
{
	Mat img_1 = Roi;
	
	int i, j;
	int a = 0;
	float hui[256] = { 0 };
	for (i = p.pt.x - p.size; i < p.pt.x + p.size; i++)
	{
		//边界检查
		if (i < 0) i = 0;
		if (i>img_1.rows - 1) continue;
		for (j = p.pt.y - p.size;j<p.pt.y+p.size; j++)
		{
			if (j < 0) j = 0;
			if (j>img_1.cols - 1) continue;
			else hui[img_1.at<uchar>(i, j)]++;
		}
	}

	for (i = 0; i < 256; i++){
		hui[i] = hui[i] / (img_1.rows*img_1.cols);
	}
	float result = 0;
	for (i = 0; i < 256; i++){
		if (hui[i] != 0){
			result = result - hui[i] * log2(hui[i]);
		}
	}
	return result;
}

//求平均半径
float avgridio(std::vector<KeyPoint> p){
	int len = p.size();
	float avgRidio = 0;
	for (int i = 0; i < len; i++){
		avgRidio += p[i].size;
	}
	return avgRidio / len;
}

//回调函数，显示点击点的坐标
void mousePoint(int event, int x, int y, int flag,void *param){
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
	{
	cout << " 坐标 "<<x<<" "<<y << endl;
	
	}
	break;
	default:
		break;
	}
}

////计算两坐标点距离
//float dis2Points(Point a, Point b){
//	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
//}



int main(int argc, char** argv)
{
	Mat img_1 = imread("d:\\project\\gray.jpg",CV_LOAD_IMAGE_COLOR);
	if (!img_1.data){
		std::cout << "Can't open" << std::endl;
	}
	SiftFeatureDetector detector;
	std::vector<KeyPoint> kp1,kp2;
	detector.detect(img_1, kp1);
	//FeaturesExtract
	SiftDescriptorExtractor extractor;
	Mat des1;
	//提取关键点
	extractor.compute(img_1,kp1,des1);
	//res为保存画点后的图形
	Mat res1;
	
	std::cout << "size of description of Img1: " << kp1.size() << std::endl;
	

	float sum,perEntropy;
	sum = 0;

	//定义存放覆盖点的容器
	std::vector<int> numcircle;
	for (int len = 0; len < kp1.size(); len++){
			perEntropy=entropy(img_1, kp1[len]);
			if (perEntropy > 0.005) kp1.pop_back();
			//circle(img_1, Point(kp1[len].pt.x, kp1[len].pt.y), kp1[len].size, Scalar(0, 255, 0));
			//cout << len << "  " << perEntropy << endl;

	}
	std::cout << "after entropy: " << kp1.size() << std::endl;
	for (int len = 0; len < kp1.size(); len++){
		numcircle.push_back(0);

		//第二层循环，计算邻域点
		for (int i = 0; i < kp1.size(); i++){
			float dis;
			if (i != len){
				dis = dis2Points(kp1[len].pt, kp1[i].pt);
				//距离小于半径，则该点覆盖的点数++
				if(dis<kp1[len].size*1.5) numcircle[len]++;
			}

		}
		//cout << len << "  " << numcircle[len]<< endl;
		if (numcircle[len]>0) {			
			//circle(img_1, Point(kp1[len].pt.x, kp1[len].pt.y), kp1[len].size, Scalar(0, 255, 0));
			kp2.push_back(kp1[len]);
		
		}
		//else
		{
			//kp1.pop_back();
		}
	}


	cout << "after filtering "<<kp2.size() << endl;

	auto kp3 = regionGrowing(img_1, kp2);
	
	for (int i = 0; i < kp3.size(); i++)
	{
		circle(img_1, kp3[i].pt, kp3[i].size, Scalar(kp3[i].class_id, kp3[i].class_id, kp3[i].class_id), -1);

	}
	//drawKeypoints(img_1, kp2, res1, Scalar::all(-1),4);
	//IplImage* transimg1 = cvCloneImage(&(IplImage)res1);

	namedWindow("Display", WINDOW_AUTOSIZE);
	//	cvShowImage("Display", transimg1);
	imshow("Display", img_1);
	//注册鼠标事件
	setMouseCallback("Display", mousePoint);


	waitKey(0);
	return 0;
}




