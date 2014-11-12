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

int numOfRegion;//��������

float dis2Points(Point a, Point b){
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}
vector<KeyPoint> regionGrowing(Mat img, vector<KeyPoint> p)
{
	vector<KeyPoint> kp3;
	float minGrowDis =20;
	int minGrayDis = 10;
	//���ӵ���Ŀ���ĻҶ�ֵ
	int seedGray, tmpGray;
	//���ӵ�
	KeyPoint seed;
	//k��������

	//��ѡ��ǿ�
	while (!p.empty())
	{
		seed = p[p.size() - 1];
		seed.class_id = numOfRegion;
		p.pop_back();
		//$$$$$$$$$$
		seedGray = img.at<uchar>(seed.pt.y, seed.pt.x);
		//�����ӵ����������
		kp3.push_back(seed);
		//���ñ����㷨
		for (int i = 0; i < p.size(); i++)
		{
			float dis;
			KeyPoint tmpPoint;
			tmpPoint = p[i];
			tmpGray = img.at<uchar>(tmpPoint.pt.y, tmpPoint.pt.x);
			//����2��������
			dis = dis2Points(seed.pt, tmpPoint.pt);
			//ͨ������ͻҶȲ��ж��Ƿ���������
			//
			if (dis < minGrowDis && abs(seedGray - tmpGray) < minGrayDis)
			{
				tmpPoint.class_id = numOfRegion;
				kp3.push_back(tmpPoint);
				p.erase(p.begin() + i);
				i--;
			}

		}

		//��������++
		numOfRegion++;
	}
	return kp3;
}

//��ֲ���
float entropy(Mat Roi, KeyPoint p)
{
	Mat img_1 = Roi;
	
	int i, j;
	int a = 0;
	float hui[256] = { 0 };
	for (i = p.pt.x - p.size; i < p.pt.x + p.size; i++)
	{
		//�߽���
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

//��ƽ���뾶
float avgridio(std::vector<KeyPoint> p){
	int len = p.size();
	float avgRidio = 0;
	for (int i = 0; i < len; i++){
		avgRidio += p[i].size;
	}
	return avgRidio / len;
}

//�ص���������ʾ����������
void mousePoint(int event, int x, int y, int flag,void *param){
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
	{
	cout << " ���� "<<x<<" "<<y << endl;
	
	}
	break;
	default:
		break;
	}
}

////��������������
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
	//��ȡ�ؼ���
	extractor.compute(img_1,kp1,des1);
	//resΪ���滭����ͼ��
	Mat res1;
	
	std::cout << "size of description of Img1: " << kp1.size() << std::endl;
	

	float sum,perEntropy;
	sum = 0;

	//�����Ÿ��ǵ������
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

		//�ڶ���ѭ�������������
		for (int i = 0; i < kp1.size(); i++){
			float dis;
			if (i != len){
				dis = dis2Points(kp1[len].pt, kp1[i].pt);
				//����С�ڰ뾶����õ㸲�ǵĵ���++
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
	//ע������¼�
	setMouseCallback("Display", mousePoint);


	waitKey(0);
	return 0;
}




