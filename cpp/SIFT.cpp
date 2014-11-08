#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;
using namespace std;

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
			if (j>img_1.cols - 1) {
				continue;
			}
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

int main(int argc, char** argv)
{
	Mat img_1 = imread("d:\\project\\gray.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	if (!img_1.data){
		std::cout << "Can't open" << std::endl;
	}
	SiftFeatureDetector detector;
	std::vector<KeyPoint> kp1, kp2;
	detector.detect(img_1, kp1);
	//FeaturesExtract
	SiftDescriptorExtractor extractor;
	Mat des1;
	//提取关键点
	extractor.compute(img_1,kp1,des1);
	//res为保存画点后的图形
	Mat res1;
	
	std::cout << "size of description of Img1: " << kp1.size() << std::endl;
	

	int numOfkp=0;//关键点数量
	float sum,perEntropy;
	sum = 0;
	for (numOfkp = 0; numOfkp < kp1.size(); numOfkp++){
		perEntropy=entropy(img_1, kp1[numOfkp]);
		if (perEntropy < 0.1) kp1.pop_back();
		cout << numOfkp << "  " << perEntropy << endl;
	}

	drawKeypoints(img_1, kp1, res1, Scalar::all(-1));
	IplImage* transimg1 = cvCloneImage(&(IplImage)res1);

	namedWindow("Dis", WINDOW_AUTOSIZE);
	cvShowImage("Dis", transimg1);

	waitKey(0);
	return 0;
}
