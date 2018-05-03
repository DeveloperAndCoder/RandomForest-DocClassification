
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>

#ifndef CLASS_HVP

#define CLASS_HVP


class HVPartitionRFC
{
	IplImage* image;	// image to be classified
	int imHeight,imWidth;
	int numCwords;
	int numPartitions; //Total number of partitions
	std::vector<int> cNumVector;
	CvMemStorage* storage;
	CvMat* points;	// N x D Matrix for descriptors of image
	CvMat *codeMat;	// K x D matrix for learned codebook
	CvSeq* imageKeypoints;
	CvSeq* imageDesc;
	int descSize;	// 64 or 128
	std::vector <float> featHVP;
	std:: vector<std::pair<float,float>> partitionBound1;
	std:: vector<std::pair<float,float>> partitionBound2;
public:
	int ComputeHVPFeatures();
	int read_codebook(const char* codebook_filename,int numCW,int descSize);
	int getDescMat( const CvSeq* model_keypoints, const CvSeq* model_descriptors, CvMat* points, int descSize );
	int ObtainSURFDesc(int descSize);
	int FindNearestCodeWordED(int index, float &mdist);
	int ObtainPartition(int h,int v);
	int ComputeHistFeatures(int index);
	std::vector<float> getHistFeat();
	HVPartitionRFC(IplImage* im, unsigned int h, unsigned int w)
	{
		image = im;
		imHeight = im->height;
		imWidth = im->width;
		imageKeypoints = 0;
		numPartitions = 0;
		//compute number of partitions
		for (int j=0; j < h ;j++)
			numPartitions = numPartitions + pow(2.0,j);
		for (int j=1; j < w ;j++)
			numPartitions = numPartitions + pow(2.0,j);

	}
	~HVPartitionRFC()
	{
		if (points != NULL)
			cvReleaseMat(&points);
		if (codeMat != NULL)
			cvReleaseMat(&codeMat);
		cvClearMemStorage(storage);
		cvReleaseMemStorage(&storage);
	}
};

#endif
