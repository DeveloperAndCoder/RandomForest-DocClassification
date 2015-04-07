
#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>

using namespace std;
using namespace cv; // OpenCV API is in the C++ "cv" namespace

#ifndef RFTrainTest

#define RFTraintest

class RandForestTrainTest{
	unsigned int numFeat;
	unsigned int numTrainSample;
	unsigned int numClasses;
	Mat featMat;
	Mat rfProximity;
	Mat data_class;
	Mat testing_data;
	CvRTParams params;
	int row_sample;
	vector<double> accuracy; 
	vector<int> class_count;
	vector <vector<int>> class_index;
	CvRTrees* rtree;
public:
	RandForestTrainTest(unsigned int nSample, unsigned int nFeat, unsigned int nClass =2, bool isProximity = false)
	{
		numFeat = nFeat;
		if (isProximity)
			numTrainSample = 2*nSample;
		else
			numTrainSample = nSample;
		row_sample = 1;
		numClasses = nClass;
	}
	RandForestTrainTest(unsigned int nFeat)
	{
		numFeat = nFeat;
	}
	double TrainTestRF(vector<int> trainIndex, vector<int> testIndex, Mat data,Mat training_classifications);
	int read_data(const char* filename );
	//int read_data_class(const char* filename);
	double TrainTestRF(const char * model_filename, bool trainflag = true);
	int do_CrossValidation(double);
	double Cross_validate(vector<int> trainIndex, vector<int> testIndex);
	int GetClassStat();
	int SetRFModel(const char *model_file);
	int	GetRFClass(double &result,double &confidence);
	int	RFClassify(const char* filename, const char *model_filename, const char *codebook_filename, unsigned int, unsigned int);
	int create_aux_data();
	int ObtainProximity(const char *);
};
#endif