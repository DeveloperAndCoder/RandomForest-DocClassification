#ifndef HVP_PARTITION
	#include "HVPartitionRFC.h"
#define HVP_PARTITION
#endif

#ifndef RAND_FOREST_TRAIN
	#include "RandForestTrain.h"
#define RAND_FOREST_TRAIN
#endif

#include <iterator>
#include <sstream>
using namespace std;

int HVPartitionRFC::getDescMat( const CvSeq* model_keypoints, const CvSeq* model_descriptors, CvMat* points, int descSize )
{
    CvSeqReader reader;
    cvStartReadSeq( model_descriptors, &reader, 0 );

    for( int i = 0; i < model_descriptors->total; i++ )
    {
        const float* mvec = (const float*)reader.ptr;
		for( int j = 0; j < descSize; j++ )
            cvmSet(points,i,j,mvec[j]);
		CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
	}
    return -1;
}

//Read codebook for computing features
int HVPartitionRFC::read_codebook(const char* codebook_filename,int numCW,int descSize)
{
	float matVal;
	int i = 0, j=0;
	numCwords = numCW;
	ifstream inputFile;
	inputFile.open (codebook_filename);
	codeMat = cvCreateMat(numCW,descSize,CV_32F);
	float *codedata = codeMat->data.fl;
	while (inputFile >> matVal)
	{
			codedata[i*descSize + j] = matVal;
			j++;
			if (j == descSize)
			{
				i++;
				j=0;
            }

	}
	return 0;
}

int HVPartitionRFC::FindNearestCodeWordED(int index, float &mdist)
{
	float minDist = 10000000;
	int k,j, minIndx;
	float absTotal;
	for (k=0; k < numCwords; k++)
	{
		absTotal = 0;
		for ( j=0; j < descSize ; j++)
            absTotal += pow(cvmGet(points,index,j)- cvmGet(codeMat,k,j),2);
		if (absTotal < minDist)
		{
			minDist = absTotal;
			minIndx = k;
		}
	}
	if (minDist > 0.1)
		minIndx = -1;
	mdist = minDist;
	return minIndx;
}

//insert the boundary co-ordinates of regions for which histogram features need to be extracted
int HVPartitionRFC::ObtainPartition(int h,int v)
{
		//partition 1 -- full image
		partitionBound1.push_back(make_pair(1,1));
		partitionBound2.push_back(make_pair(imHeight,imWidth));
		//vertical 1st half
		partitionBound1.push_back(make_pair(1,1));
		partitionBound2.push_back(make_pair(imHeight*0.5,imWidth));
		//vertical 2nd half
		partitionBound1.push_back(make_pair(imHeight*0.5,1));
		partitionBound2.push_back(make_pair(imHeight,imWidth));

		//Horizontal 1st half
		partitionBound1.push_back(make_pair(1,1));
		partitionBound2.push_back(make_pair(imHeight,imWidth*0.5));
		//Horizontal 2nd half
		partitionBound1.push_back(make_pair(1,imWidth*0.5));
		partitionBound2.push_back(make_pair(imHeight,imWidth));

		if (h >= 3)
		{
			//vertical quarters
			partitionBound1.push_back(make_pair(1,1));
			partitionBound2.push_back(make_pair(imHeight*0.25,imWidth));

			partitionBound1.push_back(make_pair(imHeight*0.25,1));
			partitionBound2.push_back(make_pair(imHeight*0.50,imWidth));

			partitionBound1.push_back(make_pair(imHeight*0.50,1));
			partitionBound2.push_back(make_pair(imHeight*0.75,imWidth));

			partitionBound1.push_back(make_pair(imHeight*0.75,1));
			partitionBound2.push_back(make_pair(imHeight,imWidth));
		}

		if (v >= 3)
		{
			//horizontal quarters
			partitionBound1.push_back(make_pair(1,1));
			partitionBound2.push_back(make_pair(imHeight,imWidth*0.25));

			partitionBound1.push_back(make_pair(1,imWidth*0.25));
			partitionBound2.push_back(make_pair(imHeight,imWidth*0.50));

			partitionBound1.push_back(make_pair(1,imWidth*0.50));
			partitionBound2.push_back(make_pair(imHeight,imWidth*0.75));

			partitionBound1.push_back(make_pair(1,imWidth*0.75));
			partitionBound2.push_back(make_pair(imHeight,imWidth));
		}
		return 0;

}
//Scan the desc. list and coun the code words
int HVPartitionRFC::ComputeHistFeatures(int index)
{
	pair<float, float> start, end1;
	start = partitionBound1[index];
	end1 = partitionBound2[index];
	float x,y;
	int unidex;
	float sum_count = 0;
	for( int i = 0; i < imageKeypoints->total; i++ )
    {
        if (cNumVector[i] == -1) // if none of the codeword was assigned
			continue;
		CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( imageKeypoints, i );
        x = r->pt.x;
        y = r->pt.y;
		if (start.first < y &&  y < end1.first && start.second < x &&  x < end1.second)
		{
			unidex = numCwords*index + cNumVector[i];
			featHVP[unidex] = featHVP[unidex] + 1;
			sum_count++;
		}

    }
	//Normalize
	int offset = numCwords*index;
	if (sum_count == 0)
		return 0;
	for (int j=0; j < numCwords ; j++)
        featHVP[offset + j] = featHVP[offset + j]/sum_count;
	return 0;
}

int HVPartitionRFC::ComputeHVPFeatures()
{
	float mindist=0;
	vector<float> mindistDesc;
	featHVP.resize(numPartitions*numCwords);
	for( int i = 0; i < imageDesc->total; i++ )
    {
        int cNumber = FindNearestCodeWordED(i,mindist); // find in the points
		cNumVector.push_back(cNumber);
		mindistDesc.push_back(mindist);
    }
	ObtainPartition(3,3);// get the boundaries so that features can be extracted
	//for each partition extract the hist features
	for ( int i = 0; i < partitionBound1.size(); i++)
        ComputeHistFeatures(i);
	return 0;
}



int HVPartitionRFC::ObtainSURFDesc(int size)
{
	descSize = size;
	cv::initModule_nonfree();
	storage = cvCreateMemStorage(0);
	CvSeq* imKeypoints = 0, *imDescriptors = 0;

	CvSURFParams params = cvSURFParams(5000, 0); // 0 for 64 dimensional
	cvExtractSURF( image, 0, &imKeypoints, &imDescriptors, storage, params );   //extracting key-points and descriptors

	points = cvCreateMat(imDescriptors->total,descSize,CV_32F);
	getDescMat( imKeypoints, imDescriptors, points, 64);

	imageKeypoints = imKeypoints;
	imageDesc = imDescriptors;

	return 0;
}

vector<float> HVPartitionRFC::getHistFeat()
{
	return featHVP;
}
