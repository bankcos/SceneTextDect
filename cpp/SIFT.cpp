
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;

int main(int argc, char** argv)
{
	Mat img_1 = imread("d:\\project\\test2.jpg",CV_LOAD_IMAGE_COLOR);
	if (!img_1.data){
		std::cout << "Can't open" << std::endl;

	}
	namedWindow("Dis", WINDOW_AUTOSIZE);
	imshow("Dis", img_1	);
	waitKey(0);
	return 0;
}