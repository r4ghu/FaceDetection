/**
Face project v0.1
FaceDetector.h
Purpose: Detects the face(s) from the given image.

@author Sri Malireddi
@version 0.1 02/05/2018
*/
#pragma once

#include "stdafx.h"

namespace FaceProject {
	/*
	An OpenCV Wrapper class to detect face(s)
	*/
	class FaceDetector {
	public:
		/**
		Constructor to initialize the FaceDetector object with path to cascadeFile

		@param cascadeFile The path to face cascade file.
		@usage FaceProject::FaceDetector detector(<path-to-file>)
		*/
		FaceDetector(std::string cascadeFile) {
			this->Init(cascadeFile);
		}

		/**
		Constructor to initialize the FaceDetector object with path to cascadeFile,
		and other important variables.

		@param cascadeFile The path to face cascade file.
		@param imageScale The variable used to rescale the image.
		@param scaleFactor Parameter specifying how much the image size is reduced at each image scale
		@param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have to retain it.
		@param minSize Minimum possible object size.
		@usage FaceProject::FaceDetector detector(<path-to-file>,...variables)
		*/
		FaceDetector(std::string cascadeFile,
			double imageScale,
			double scaleFactor,
			int minNeighbors,
			int minSize) {
			this->Init(cascadeFile, imageScale, scaleFactor, minNeighbors, minSize);
		}

		FaceDetector(std::string prototxt, std::string caffeModel) {
			this->Init(prototxt, caffeModel);
		}

		/**
		Returns the closest face to the camera in cv::Rect format.

		@param img The image to detect face.
		@return The cv::Rect of the closest face in the image
		*/
		cv::Rect DetectFace(cv::Mat& img) {

			std::vector<cv::Rect> faces = DetectFaces(img);


			if (faces.size() == 0) return cv::Rect(0, 0, 0, 0);

			int maxFaceArea = -1; cv::Rect retFace;
			for (auto face : faces) {
				if (maxFaceArea < face.width * face.height) {
					maxFaceArea = face.width * face.height;

					retFace.x = face.x * _imageScale;
					retFace.y = face.y * _imageScale;
					retFace.width = face.width * _imageScale;
					retFace.height = face.height * _imageScale;
				}
			}
			return retFace;
		}

		/**
		Returns all the faces in the image in std::vector<cv::Rect> format.

		@param img The image to detect faces.
		@return The std::vector<cv::Rect> of all the faces in the image.
		*/
		std::vector<cv::Rect> DetectFaces(cv::Mat& img) {

			assert(img.type() % 8 == 0); // Check if image is of type CV_8U

			cv::Mat smallImg = PreProcess(img);

			std::vector<cv::Rect> faces;
			if (_method == METHOD::CASCADE) {
				_faceCascade.detectMultiScale(smallImg, faces, _scaleFactor, _minNeighbors, 0, _minSize);
			}
			else if (_method == METHOD::DNN) {
				_faceSSDNet.setInput(smallImg);
				cv::Mat detections = _faceSSDNet.forward();

				// Post-process
				float* data = (float*)detections.data;
				for (size_t i = 0; i < detections.total(); i += 7) {
					float confidence = data[i + 2];
					if (confidence > _confThreshold) {
						int xLeftBottom = (int)data[i + 3] * img.cols;
						int yLeftBottom = (int)data[i + 4] * img.rows;
						int xRightTop = (int)data[i + 5] * img.cols;
						int yRightTop = (int)data[i + 6] * img.rows;

						faces.push_back(cv::Rect(cv::Point(xLeftBottom, yLeftBottom), cv::Point(xRightTop, yRightTop)));
					}
				}
			}

			return faces;
		}
	private:
		std::string _cascadeFile;
		double _imageScale, _scaleFactor;
		int _minNeighbors;
		cv::Size _minSize;
		cv::CascadeClassifier _faceCascade;
		cv::dnn::Net _faceSSDNet;
		float _confThreshold = 0.5f;
		
		enum METHOD {
			CASCADE = 0,
			DNN = 1
		};

		METHOD _method = METHOD::CASCADE;
		
		

		/**
		Function to initialize all the variables of cascade classifier.

		@param cascadeFile The path to face cascade file.
		@param imageScale The variable used to rescale the image.
		@param scaleFactor Parameter specifying how much the image size is reduced at each image scale
		@param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have to retain it.
		@param minSize Minimum possible object size.
		*/
		void Init(const std::string cascadeFile,
			const double imageScale = 1.3,
			const double scaleFactor = 1.1,
			const int minNeighbors = 2,
			const int minSize = 30) {
			_cascadeFile = cascadeFile;
			_imageScale = imageScale;
			_scaleFactor = scaleFactor;
			_minNeighbors = minNeighbors;
			_minSize = cv::Size(minSize, minSize);

			if (!_faceCascade.load(_cascadeFile)) {
				std::cerr << "ERROR: Load not success. Face cascade file not found!!!" << std::endl;
			}

			_method = METHOD::CASCADE;
		}

		void Init(const std::string prototxtFile,
			const std::string caffeModelFile) {
			_faceSSDNet = cv::dnn::readNetFromCaffe(prototxtFile, caffeModelFile);
			_imageScale = 1.0;
			_method = METHOD::DNN;
		}

		/**
		Returns the pre-processed image upon which the detector is run

		@param img The image to detect faces.
		@return The pre-processed image.
		*/
		cv::Mat PreProcess(cv::Mat& img) {

			if (_method == METHOD::CASCADE) {
				// Convert to gray scale if needed
				cv::Mat gray;
				if (img.channels() == 1) gray = img.clone();
				else cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

				// Resize image as per the given scale and pre-process
				cv::Mat smallImg;
				int w = (int)img.cols / _imageScale;
				int h = (int)img.rows / _imageScale;
				if (_imageScale != 1.0) {
					cv::resize(gray, smallImg, cv::Size(w, h), 0, 0, cv::INTER_AREA);
				}
				else {
					smallImg = gray;
				}
				cv::equalizeHist(smallImg, smallImg);
				return smallImg;
			}

			else if (_method == METHOD::DNN) {
				cv::Mat inputBlob = cv::dnn::blobFromImage(img, 1.0, 
													cv::Size(300, 300), 
													cv::Scalar(104.0, 177.0, 123.0), 
													false, false);
				return inputBlob;
			}
			
		}
	};
}