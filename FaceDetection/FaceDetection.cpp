// FaceDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FaceDetector.h"

// TODO: Change this to your project location
#define SOLUTION_DIR "C:/Users/srira/Documents/GitHub/FaceDetection/"

#define DNN

int main()
{
	//Initialize webcam
	cv::VideoCapture cap(0);
#ifdef DNN
	std::string caffeModelFile = SOLUTION_DIR + std::string("model/res10_300x300_ssd_iter_140000.caffemodel");
	std::string prototxtFile = SOLUTION_DIR + std::string("model/deploy.prototxt");
	FaceProject::FaceDetector faceDetector(prototxtFile, caffeModelFile);
#else
	std::string path = SOLUTION_DIR + std::string("model/haarcascade_profileface.xml");
	FaceProject::FaceDetector faceDetector(path);
#endif // DNN

	
	cv::Mat frame;
	while (1) {
		bool success = cap.read(frame);

		if (!success) continue;

		cv::Rect face = faceDetector.DetectFace(frame);
		cv::rectangle(frame, face, cv::Scalar(255, 255, 255), 2);
		cv::imshow("Image", frame);

		char k = cv::waitKey(1);
		if (k == 27) {
			break;
		}
	}
	return 0;
}

