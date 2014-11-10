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
	//ͳ�ƻҶȼ���ÿ������������ͼ�еĸ���
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			pixelCount[img.at<uchar>(i, j)]++;
		}
	}

	//����ÿ�׻Ҷ�������ͼ���еı���
	for (int i = 0; i < 256; i++)
	{
		pixelPro[i] = (float)(pixelCount[i]) / (float)(pixelSum);
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

		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp= c0 = c1 = 0;

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

int main()
{
	Mat src_gray,src,src_bin;

	src= imread("d:\\project\\test.jpg");
	cvtColor(src, src_gray, CV_RGB2GRAY);

	//���������ֵ
	int thresholdvalue = ostu(src_gray);
	//��˹�˲�
	//GaussianBlur(src_gray, src_gray,Size(5,5), 0);
	//��ͼ���ֵ��
	//auto oksd = threshold(src_gray, src_bin, thresholdvalue, 255, CV_THRESH_BINARY );
	auto oksd=threshold(src_gray,src_bin, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU);

	cvNamedWindow("binary");
	imshow("binary", src_bin);

	waitKey( );


	return 0;
}