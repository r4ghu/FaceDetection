// FaceDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FaceDetector.h"

// TODO: Change this to your project location
#define SOLUTION_DIR "C:/src/FaceDetector/"

int main()
{
	//Initialize webcam
	cv::VideoCapture cap(0);
	std::string path = SOLUTION_DIR + std::string("model/haarcascade_frontalface_alt.xml");

	FaceProject::FaceDetector faceDetector(path);

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

