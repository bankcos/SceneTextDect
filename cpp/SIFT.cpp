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
struct recRange
{
	int x1, y1, x2, y2;
};

int numOfRegion;//聚类数量
vector<KeyPoint> kp3;
bool drawing;
int mouseX, mouseY;
Mat img_0;
vector<recRange>  myRange;
Mat sampleFeature, sampleLabel;


float dis2Points(Point a, Point b){
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

vector<KeyPoint> regionGrowing(Mat img, vector<KeyPoint> p)
{
	//临时容器
	vector<KeyPoint> kp;
	float minGrowDis = 15;
	int minGrayDis = 9;
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
		kp.push_back(seed);

		//此双重循环后剩余点需重新聚合
		for (int i = kp.size()-1; i < kp.size(); i++)
		{
			//按顺序取kp中的元素，作为种子点
			seed = kp[i];
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
					kp.push_back(tmpPoint);
					p.erase(p.begin() + j);
					j--;
				}

			}
		}
		//聚类种类++
		numOfRegion++;
	}
	return kp;
}

int flagKeypoint(vector<KeyPoint> kp)
{
	for (int i = 0; i < kp.size(); i++)
	{

		int c;
		c = 1;
		Mat tmpLabel(1, 1, CV_16UC1, c);
		sampleLabel.push_back(tmpLabel);
		
	}
	cout << sampleLabel << endl;
	cout << sampleLabel.size() << endl;
	return 0;
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

int findPoint(int x, int y,vector<KeyPoint> p)
{
	float dis ,mindis=999999999;
	int minIndex;
	int i;
	for ( i = 0; i < p.size()-1; i++)
	{
		dis = dis2Points(Point(x, y), p[i].pt);
		if (dis < mindis)
		{
			mindis = dis;
			minIndex = i;
		}
	}
	return minIndex;
}

//回调函数，显示点击点的坐标
void mousePoint(int event, int x, int y, int flag,void *param){
	Mat tmpimg;
	recRange tmpRag;
	img_0.copyTo(tmpimg);
	switch (event)
	{	
		//鼠标左键开始选择区域
		case CV_EVENT_LBUTTONDOWN:
		{
			drawing = true;
			mouseX = x; mouseY = y;

		}
			break;
	case CV_EVENT_MOUSEMOVE:
		if (drawing == true && flag == 1) rectangle(tmpimg, Point(mouseX, mouseY), Point(x, y), (255, 0, 255), 1);
		imshow("Display", tmpimg);
		break;

	case CV_EVENT_LBUTTONUP:
		drawing = false;
		tmpRag.x1 = mouseX; tmpRag.y1 = mouseY;
		tmpRag.x2 = x; tmpRag.y2 = y;
		myRange.push_back(tmpRag);
		break;
	//按鼠标中键取消上次的选点
	case CV_EVENT_MBUTTONDBLCLK:
		if (myRange.size() > 0) myRange.pop_back();
		break;
	case CV_EVENT_RBUTTONDOWN:
		cout << "聚类 "<<kp3[findPoint(x,y,kp3)].class_id<< endl;
		break;
	default:
		break;
	}
}



////计算两坐标点距离
//float dis2Points(Point a, Point b){
//	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
//}



int bProgess(String path)
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


	auto e1 = getTickCount();
	Mat img_1 = imread(path,CV_LOAD_IMAGE_GRAYSCALE);

	if (!img_1.data){
		std::cout << "Can't open" << std::endl;
	}
	resize(img_1, img_1, Size(640, 480), INTER_CUBIC);

	SiftFeatureDetector detector;
	std::vector<KeyPoint> kp1,kp2;
	detector.detect(img_1, kp1);
	//FeaturesExtract
	SiftDescriptorExtractor extractor;

	std::cout << "Totel: " << kp1.size() << std::endl;
/**************************
	熵计算
***************************/
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
/**************************
剔除相邻点
***************************/
	for (int len = 0; len < kp1.size(); len++)
	{
		numcircle.push_back(0);

		//第二层循环，计算邻域点
		for (int i = 0; i < kp1.size(); i++)
		{
			float dis;
			if (i != len){
				dis = dis2Points(kp1[len].pt, kp1[i].pt);
				//距离小于半径，则该点覆盖的点数++
				if (dis<kp1[len].size*1.5) numcircle[len]++;
			}

		}
		//cout << len << "  " << numcircle[len]<< endl;
		if (numcircle[len]>0) 
		{			
			//circle(img_1, Point(kp1[len].pt.x, kp1[len].pt.y), kp1[len].size, Scalar(0, 255, 0));
			kp2.push_back(kp1[len]);
		
		}
		//else
		{
			//kp1.pop_back();
		}
	}
	cout << "after filtering " << kp2.size() << endl;

	Mat des2;
	extractor.compute(img_1, kp2, des2);
/**************************
		区域生长
**************************/
	
	//kp3 = regionGrowing(img_1, kp2);
	
	//for (int i = 0; i < kp3.size(); i++)
	//{
	//	circle(img_1, kp3[i].pt, kp3[i].size, Scalar(0, kp3[i].class_id, 0), -1);
	//}

	//drawKeypoints(img_1, kp3, img_1, Scalar::all(-1), 4);
	img_0 = img_1;
/**************************
	显示图片
**************************/

	
	//	cvShowImage("Display", transimg1);


	imshow("Display", img_1);
                                                                                    
//	int thresholdvelue = ostuKeypoint(img_1,kp2);

	//Mat src_bin;
	//auto oksd = threshold(img_1, src_bin, thresholdvelue, 255, CV_THRESH_BINARY);


	//drawKeypoints(img_1, kp1, res1, Scalar::all(-1),4);
//	IplImage* transimg1 = cvCloneImage(&(IplImage)res1);


	//注册鼠标事件
	setMouseCallback("Display", mousePoint);

	//drawKeypoints(img_1, kp2, res1, Scalar::all(-1), 4);
	//IplImage* transimg1 = cvCloneImage(&(IplImage)res1);
	//namedWindow("Display", WINDOW_AUTOSIZE);
	//cvShowImage("Display", transimg1);
	//imshow("Display", src_bin);
	auto e2 = getTickCount();
	cout << "用时" << (e2 - e1)/getTickFrequency() << "s" << endl;
	waitKey(0);

	for (int i = 0; i < kp2.size(); i++)
	{	
		float c=-1;
		Point a = kp2[i].pt;
		if (myRange.empty() == true)
		{
			cout << "没有选定的点" << endl;
			break;
		}
		else
		{
			for (int j = 0; j < myRange.size(); j++)
			{
				if ((a.x > myRange[myRange.size() - 1].x1) &&
					(a.x < myRange[myRange.size() - 1].x2) &&
					(a.y > myRange[myRange.size() - 1].y1) &&
					(a.y > myRange[myRange.size() - 1].y2)
					)
					c = 1;
			}
		}

		Mat tmpLabel(1, 1, CV_32FC1, c);
		sampleLabel.push_back(tmpLabel);
	}

	sampleFeature.push_back(des2);
	//清空myRange
	myRange.clear();

	return 0;
}

int main()
{
	sampleFeature.create(0, 2, CV_32FC1);
	sampleLabel.create(0, 1, CV_32FC1);
	cout << sampleLabel << endl;
	namedWindow("Display", WINDOW_AUTOSIZE);
	string samplePath = "d:\\project\\traindata\\";
	string path;
	char sampleName[256];
	Mat a;
	for (int i = 100; i < 101; i++)
	{
		memset(sampleName, '\0', sizeof(char)* 256);
//		sprintf_s(sampleName, "%04i.jpg", i);
		sprintf_s(sampleName, "%d.jpg", i);
		//文件路径
		path = samplePath + sampleName;

		auto c = bProgess(path);

	}
	FileStorage fs("d:\\project\\feature.xml", FileStorage::WRITE);
	fs << "samplefeature" << sampleFeature;
	fs.release();
	FileStorage ys("d:\\project\\label.xml", FileStorage::WRITE);
	ys << "samplelabel" << sampleLabel;
	ys.release();


	return 0;
}


