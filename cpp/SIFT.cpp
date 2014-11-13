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
using namespace std;


int numOfRegion;//聚类数量

float dis2Points(Point a, Point b){
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}
/*
vector<KeyPoint> regionGrowing(Mat img, vector<KeyPoint> p)
{
	vector<KeyPoint> kp3;
	float minGrowDis =20;
	int minGrayDis = 10;
	//种子点与目标点的灰度值
	int seedGray, tmpGray;
	//种子点
	KeyPoint seed;

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
*/
vector<KeyPoint> regionGrowing(Mat img, vector<KeyPoint> p)
{
	//临时容器
	vector<KeyPoint> kp3;
	float minGrowDis = 20;
	int minGrayDis = 10;
	//种子点与目标点的灰度值
	int seedGray, tmpGray;
	//种子点
	KeyPoint seed;

	//候选点非空
	while (!p.empty())
	{
		seed = p[p.size() - 1];
		seed.class_id = numOfRegion;
		p.pop_back();
		kp3.push_back(seed);

		//此双重循环后剩余点需重新聚合
		for (int i = kp3.size()-1; i < kp3.size(); i++)
		{
			//按顺序取kp3中的元素，作为种子点
			seed = kp3[i];
			//$$$$$$$$$$
			seedGray = img.at<uchar>(seed.pt.y, seed.pt.x);
			for (int j = 0; j < p.size();j++)
			{
				float dis;
				KeyPoint tmpPoint;
				tmpPoint = p[j];
				tmpGray = img.at<uchar>(tmpPoint.pt.y, tmpPoint.pt.x);
				//计算2两点间距离
				dis = dis2Points(seed.pt, tmpPoint.pt);
				//通过距离和灰度差判断是否是我族类
				if (dis < minGrowDis && abs(seedGray - tmpGray) < minGrayDis)
				{
					tmpPoint.class_id = numOfRegion;
					kp3.push_back(tmpPoint);
					p.erase(p.begin() + j);
					j--;
				}

			}
		}
		//聚类种类++
		numOfRegion++;
	}
	return kp3;
}


int ostuKeypoint(Mat img,std::vector<KeyPoint> p){
	assert(NULL != img.data);

	KeyPoint p1;
	int width = img.cols;
	int height = img.rows;
	int pixelCount[256] = { 0 };
	float pixelPro[256] = { 0 };
	//总数为关键点个数
	int pixelSum = p.size();
	int threshold = 0;

	//统计灰度级中每个像素在整幅图中的个数
	for (int i = 0; i < pixelSum; i++){
		p1 = p[i];

		//在关键点处的灰度值
		pixelCount[img.at<uchar>(p1.pt.y,p1.pt.x)]++;

	}
	float tmp1 = 0;
	//计算每阶灰度在关键点中的比例
	for (int i = 0; i < 256; i++)
	{
		pixelPro[i] = (float)(pixelCount[i]) / (float)(pixelSum);
		tmp1 += pixelPro[i];
		cout << tmp1 << endl;
	}

	//经典ostu算法，得到前景和背景的分割
	//遍历灰度级[0,255]，计算出方差的最大灰度值，为最佳阈值
	/* 对于一幅图像，设当前景与背景的分割阈值为t时，前景点占图像比例为w0，
	均值为u0，背景点占图像比例为w1，均值为u1。则整个图像的均值为u = w0*u0+w1*u1。
	建立目标函数g(t)=w0*(u0-u)^2+w1*(u1-u)^2，g(t)就是当分割阈值为t时的类间方差表达式。
	OTSU算法使得g(t)取得全局最大值，当g(t)为最大时所对应的t称为最佳阈值。	*/
	float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax, c0, c1 = 0;
	deltaMax = 0;
	for (int i = 0; i < 256; i++){

		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = c0 = c1 = 0;

		for (int j = 0; j < 256; j++)
		{
			if (j <= i) //前景部分
			{
				//以i为阈值分类，第一类总的概率
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];

			}
			else       //背景部分
			{
				//以i为阈值分类，第二类总的概率
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];

			}
		}

		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		u = u0tmp + u1tmp;  //平均灰度
		//		cout << i<<"  "<<u0 << "  " << u1 << "   "<<u << endl;
		//	cout << i << "  " << u << endl;
		//计算间类方差
		deltaTmp = w0 * (u0 - u)*(u0 - u) + w1 * (u1 - u)*(u1 - u);
		//找出最大值
		if (deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}

	return threshold;
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
	/*
	vector<char> p;
	for (int i = 0; i < 3; i++){
		p.push_back(i);
	}

	for (int i = 0; i < p.size(); i++){
		p.push_back(1);
		cout << i << endl;

	}
*/


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
		circle(img_1, kp3[i].pt, kp3[i].size, Scalar(0, kp3[i].class_id, 0), -1);

	}
	//drawKeypoints(img_1, kp2, res1, Scalar::all(-1),4);
	//IplImage* transimg1 = cvCloneImage(&(IplImage)res1);

	namedWindow("Display", WINDOW_AUTOSIZE);
	//	cvShowImage("Display", transimg1);
	imshow("Display", img_1);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
	int thresholdvelue = ostuKeypoint(img_1,kp2);

	//Mat src_bin;
	//auto oksd = threshold(img_1, src_bin, thresholdvelue, 255, CV_THRESH_BINARY);


	//drawKeypoints(img_1, kp1, res1, Scalar::all(-1),4);
	IplImage* transimg1 = cvCloneImage(&(IplImage)res1);

//	namedWindow("Display", WINDOW_AUTOSIZE);
//	cvShowImage("Display", transimg1);
	//imshow("Display", src_bin);

	//注册鼠标事件
	setMouseCallback("Display", mousePoint);


	waitKey(0);
	return 0;
}




