#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int ostu(Mat img){
	assert(NULL != img.data);

	int width = img.cols;
	int height = img.rows;
	int pixelCount[256] = {0};
	float pixelPro[256] = {0};
	int pixelSum = width*height, threshold = 0;

	uchar *data = img.data;
	//统计灰度级中每个像素在整幅图中的个数
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			pixelCount[img.at<uchar>(i, j)]++;
		}
	}

	//计算每阶灰度在整幅图像中的比例
	for (int i = 0; i < 256; i++)
	{
		pixelPro[i] = (float)(pixelCount[i]) / (float)(pixelSum);
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

		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp= c0 = c1 = 0;

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

int main()
{
	Mat src_gray,src,src_bin;

	src= imread("d:\\project\\test.jpg");
	cvtColor(src, src_gray, CV_RGB2GRAY);

	//计算最佳阈值
	int thresholdvalue = ostu(src_gray);
	//高斯滤波
	//GaussianBlur(src_gray, src_gray,Size(5,5), 0);
	//对图像二值化
	//auto oksd = threshold(src_gray, src_bin, thresholdvalue, 255, CV_THRESH_BINARY );
	auto oksd=threshold(src_gray,src_bin, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU);

	cvNamedWindow("binary");
	imshow("binary", src_bin);

	waitKey( );


	return 0;
}