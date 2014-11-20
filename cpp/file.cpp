#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace cv;
using namespace std;



void main()
{
/*	
	string samplePath = "d:\\project\\MSRG\\MSRG_";
	string path;
	char sampleName[256];
	for (int i = 1; i < 200; i++)
	{
		memset(sampleName, '\0', sizeof(char)* 256);
		sprintf(sampleName, "%04i.jpg", i);
		path = samplePath + sampleName;
		cout << path << endl;
	}
*/
	FileStorage fs("d:\\project\\test.xml", FileStorage::WRITE);



	fs.release();


	system("pause");
}