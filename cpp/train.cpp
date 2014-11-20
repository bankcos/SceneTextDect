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

int showImg(Mat img)
{

	namedWindow("Display", WINDOW_AUTOSIZE);
	//	cvShowImage("Display", transimg1);

	resize(img, img, Size(640, 480), INTER_CUBIC);
	imshow("Display", img);
	waitKey(0);
	return 0;
}


Mat  tSIFT(String path)
{
	Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	//特征点描述符
	Mat des;
	if (!img.data){
		std::cout << "Can't open" << std::endl;
		system("Pause");
		exit(0);
	}

	SiftFeatureDetector detector;
	std::vector<KeyPoint> tSIFTkp;
	detector.detect(img, tSIFTkp);
	Mat img1;
	drawKeypoints(img, tSIFTkp, img1, Scalar::all(-1), 4);
	//FeaturesExtract
	SiftDescriptorExtractor extractor;
	//提取特征向量
	extractor.compute(img,tSIFTkp,des);

	showImg(img1);

	return des;
}


Mat fileRead(String path,String name)
{
	Mat a;
	FileStorage fs(path, FileStorage::READ);
	fs[name] >> a;
	fs.release();
	return a;
}

void mySvm(Mat feature,Mat label)
{
	// 设置SVM参数
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	cout << "开始训练" << endl;
	// 对SVM进行训练
	CvSVM SVM;
	auto a=SVM.train(feature, label, Mat(), Mat(), params);


}


int main()
{
	Mat sampleFeature, sampleLable;
	sampleFeature = fileRead("d:\\project\\feature.xml","samplefeature");
	sampleLable = fileRead("d:\\project\\label.xml", "samplelabel");

	mySvm(sampleFeature, sampleLable);
	return 0;
}

