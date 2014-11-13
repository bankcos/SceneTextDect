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


int numOfRegion;//��������

float dis2Points(Point a, Point b){
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}
/*
vector<KeyPoint> regionGrowing(Mat img, vector<KeyPoint> p)
{
	vector<KeyPoint> kp3;
	float minGrowDis =20;
	int minGrayDis = 10;
	//���ӵ���Ŀ���ĻҶ�ֵ
	int seedGray, tmpGray;
	//���ӵ�
	KeyPoint seed;

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
*/
vector<KeyPoint> regionGrowing(Mat img, vector<KeyPoint> p)
{
	//��ʱ����
	vector<KeyPoint> kp3;
	float minGrowDis = 20;
	int minGrayDis = 10;
	//���ӵ���Ŀ���ĻҶ�ֵ
	int seedGray, tmpGray;
	//���ӵ�
	KeyPoint seed;

	//��ѡ��ǿ�
	while (!p.empty())
	{
		seed = p[p.size() - 1];
		seed.class_id = numOfRegion;
		p.pop_back();
		kp3.push_back(seed);

		//��˫��ѭ����ʣ��������¾ۺ�
		for (int i = kp3.size()-1; i < kp3.size(); i++)
		{
			//��˳��ȡkp3�е�Ԫ�أ���Ϊ���ӵ�
			seed = kp3[i];
			//$$$$$$$$$$
			seedGray = img.at<uchar>(seed.pt.y, seed.pt.x);
			for (int j = 0; j < p.size();j++)
			{
				float dis;
				KeyPoint tmpPoint;
				tmpPoint = p[j];
				tmpGray = img.at<uchar>(tmpPoint.pt.y, tmpPoint.pt.x);
				//����2��������
				dis = dis2Points(seed.pt, tmpPoint.pt);
				//ͨ������ͻҶȲ��ж��Ƿ���������
				if (dis < minGrowDis && abs(seedGray - tmpGray) < minGrayDis)
				{
					tmpPoint.class_id = numOfRegion;
					kp3.push_back(tmpPoint);
					p.erase(p.begin() + j);
					j--;
				}

			}
		}
		//��������++
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
	//����Ϊ�ؼ������
	int pixelSum = p.size();
	int threshold = 0;

	//ͳ�ƻҶȼ���ÿ������������ͼ�еĸ���
	for (int i = 0; i < pixelSum; i++){
		p1 = p[i];

		//�ڹؼ��㴦�ĻҶ�ֵ
		pixelCount[img.at<uchar>(p1.pt.y,p1.pt.x)]++;

	}
	float tmp1 = 0;
	//����ÿ�׻Ҷ��ڹؼ����еı���
	for (int i = 0; i < 256; i++)
	{
		pixelPro[i] = (float)(pixelCount[i]) / (float)(pixelSum);
		tmp1 += pixelPro[i];
		cout << tmp1 << endl;
	}

	//����ostu�㷨���õ�ǰ���ͱ����ķָ�
	//�����Ҷȼ�[0,255]���������������Ҷ�ֵ��Ϊ�����ֵ
	/* ����һ��ͼ���赱ǰ���뱳���ķָ���ֵΪtʱ��ǰ����ռͼ�����Ϊw0��
	��ֵΪu0��������ռͼ�����Ϊw1����ֵΪu1��������ͼ��ľ�ֵΪu = w0*u0+w1*u1��
	����Ŀ�꺯��g(t)=w0*(u0-u)^2+w1*(u1-u)^2��g(t)���ǵ��ָ���ֵΪtʱ����䷽����ʽ��
	OTSU�㷨ʹ��g(t)ȡ��ȫ�����ֵ����g(t)Ϊ���ʱ����Ӧ��t��Ϊ�����ֵ��	*/
	float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax, c0, c1 = 0;
	deltaMax = 0;
	for (int i = 0; i < 256; i++){

		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = c0 = c1 = 0;

		for (int j = 0; j < 256; j++)
		{
			if (j <= i) //ǰ������
			{
				//��iΪ��ֵ���࣬��һ���ܵĸ���
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];

			}
			else       //��������
			{
				//��iΪ��ֵ���࣬�ڶ����ܵĸ���
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];

			}
		}

		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		u = u0tmp + u1tmp;  //ƽ���Ҷ�
		//		cout << i<<"  "<<u0 << "  " << u1 << "   "<<u << endl;
		//	cout << i << "  " << u << endl;
		//������෽��
		deltaTmp = w0 * (u0 - u)*(u0 - u) + w1 * (u1 - u)*(u1 - u);
		//�ҳ����ֵ
		if (deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}

	return threshold;
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

	//ע������¼�
	setMouseCallback("Display", mousePoint);


	waitKey(0);
	return 0;
}




