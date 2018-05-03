
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"


#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iterator>
#include <sstream>

using namespace cv;
using namespace std;

class ImageBasedCodeBook
{
	Mat points;
	CvSeq* imageKeypoints;
	CvSeq* imageDescriptors;
	Mat labels;
	Mat centers;
	unsigned int clusterCount;
	unsigned int offset;
	unsigned int max_sample;
	unsigned int max_flag;
public:
	ImageBasedCodeBook()
	{
		offset = 0;
		max_sample = 1000000;
		max_flag = 0;

	}
	int CreateCodeBook(const char* image_filename, const char* codebook_filename, int clusterCount = 100);
	int addDescMatrix( const CvSeq* model_keypoints, const CvSeq* model_descriptors,  int descSize = 64);
};

int ImageBasedCodeBook::addDescMatrix( const CvSeq* model_keypoints, const CvSeq* model_descriptors, int descSize)
{
    if ( offset < max_sample)
		points.resize(offset + model_descriptors->total,0); // resize the matrix, add more rows
	else
	{
		max_flag = 1;
		return 0;
	}
    CvSeqReader reader;
    cvStartReadSeq( model_descriptors, &reader, 0 );

    for( int i = 0; i < model_descriptors->total; i++ )
    {
        const float* mvec = (const float*)reader.ptr;
		//copy the vector into matrix
		for( int j = 0; j < descSize; j++ )
		 	 points.at<float>(i + offset,j) = mvec[j];
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
	}
	offset += model_descriptors->total;
    return 0;
}

//Read text-file for image-names and create SURF based codebook, save the codebook as text file
int ImageBasedCodeBook::CreateCodeBook(const char* image_filename, const char* codebook_filename, int nC)
{
	int row, col;
	string fname;
	ifstream inputFile;
	clusterCount = nC; // number of clusters
	inputFile.open (image_filename); // file containing images names
	IplImage* image ;
	cv::initModule_nonfree();
	CvSURFParams params = cvSURFParams(5000, 0); // 0 for 64 dimensional
	points = Mat(1,64,CV_32F);
	int i = 0;
	int attempts = 3;
	while (inputFile >> fname)
	{
	    i++;
		cout<<fname;
		image = cvLoadImage( fname.c_str(), CV_LOAD_IMAGE_GRAYSCALE );
		if(!image )
			fprintf( stderr, "Can not load %s\n", fname.c_str());

        CvMemStorage* storage = cvCreateMemStorage(0);
		cvExtractSURF( image, 0, &imageKeypoints, &imageDescriptors, storage, params ); //extracting key-points and descriptors
		printf("\nObject Descriptors: %d\n", imageDescriptors->total);
		addDescMatrix( imageKeypoints, imageDescriptors);
		cvClearMemStorage(storage);
		cvReleaseMemStorage(&storage);  //deallocation
		cvReleaseImage(&image); //deallocation
		if (max_flag == 1)
            break;
	}

    cout << "\n\nImages loaded = " << i << endl;
	labels = Mat(offset, 1, CV_32SC1 );
	centers = Mat(clusterCount,64,CV_32F);

	kmeans(points, clusterCount, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 0.0001, 10000), attempts, cv::KMEANS_PP_CENTERS,centers );

	std::ofstream file(codebook_filename);
	  if (file.is_open())
	  {
		  for (row = 0; row < clusterCount ; row++)
		  {
			  for (col = 0; col < 64; col++)
			  {
				  file << centers.at<float>(row,col);
				  file << " ";
			  }
			  file << "\n";
		  }
	  }
	  file.close();
	  return 0;
}
