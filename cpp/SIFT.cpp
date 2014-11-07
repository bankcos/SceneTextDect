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
		for (j = p.pt.y - p.size;j<p.pt.y+p.size; j++)
		{
			hui[img_1.at<uchar>(i, j)]++;
			//			cout << a++ << endl;
			//img_1.at<uchar>(i, j)[0] = 0;
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
	Mat M(7, 7, CV_32FC2, Scalar(1, 3));
	Mat img_1 = imread("d:\\project\\smallgray.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	if (!img_1.data){
		std::cout << "Can't open" << std::endl;

	}
	SiftFeatureDetector detector;
	std::vector<KeyPoint> kp1, kp2;
	detector.detect(img_1, kp1);
	//FeaturesExtract
	SiftDescriptorExtractor extractor;
	Mat des1;
	extractor.compute(img_1,kp1,des1);

	Mat res1;
	int drawmode = DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
	drawKeypoints(img_1, kp1, res1, Scalar::all(-1), 4);
	std::cout << "size of description of Img1: " << kp1.size() << std::endl;
	
	IplImage* transimg1 = cvCloneImage(&(IplImage)res1);
//	namedWindow("Dis", WINDOW_AUTOSIZE);
	cvShowImage("Dis", transimg1);
//	imshow("Dis", img_1	);
	cout << entropy(img_1,kp1[9]);
//	entropy(des1, img_1.size(), 256);
	waitKey(0);
	return 0;
}
