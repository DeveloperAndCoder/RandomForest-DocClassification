// Implementation of Horizontal-Vertical partitioning class for structural similarity based document classification
// Author: Jayant Kumar, jayant@umiacs.umd.edu

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
    int length = (int)(model_descriptors->elem_size/sizeof(float));
    int i,j;
	
    CvSeqReader reader, kreader;
    //cvStartReadSeq( model_keypoints, &kreader, 0 );
    cvStartReadSeq( model_descriptors, &reader, 0 );

    for( i = 0; i < model_descriptors->total; i++ )
    {
        //const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
        const float* mvec = (const float*)reader.ptr;
		//copy the vector into matrix
		 for( j = 0; j < descSize; j++ )
		 {
			 cvmSet(points,i,j,mvec[j]);
		 }
		 //CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
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
			//std::cout <<matVal << "  ";
			if (j == descSize)
			{
				i++;
				j=0;
				//std::cout << std::endl;
			}
			
	}
	return 0;
}

int HVPartitionRFC::FindNearestCodeWord(int index, float &mdist)
{
	float minDist = 10000000;
	int k,j, minIndx;
	float absTotal;
	for (k=0; k < numCwords; k++)
	{
		absTotal = 0;
		for ( j=0; j < descSize ; j++)
		{
				//cout<<cvmGet(points,k,j)<<"\t";
				absTotal = absTotal + abs(cvmGet(points,index,j)- cvmGet(codeMat,k,j));
		}
		if (absTotal < minDist)
		{
			minDist = absTotal;
			minIndx = k;
		}
	}
	//cout<<"\n Min dist: "<<minDist;
	if (minDist > 1.5)
		minIndx=-1;
	mdist = minDist;
	return minIndx;
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
		{
				//cout<<cvmGet(points,k,j)<<"\t";
				absTotal = absTotal + pow(cvmGet(points,index,j)- cvmGet(codeMat,k,j),2);
		}
		if (absTotal < minDist)
		{
			minDist = absTotal;
			minIndx = k;
		}
	}
	//cout<<"\n Min dist: "<<minDist;
	if (minDist > 0.1)
		minIndx = -1;
	mdist = minDist;
	return minIndx;
}


//return partitions which needs to be updated
int HVPartitionRFC::GetUpdateIndices(int x, int y, vector<int> &update)
{
	update.push_back(1); //first partition - GLOBAL

	//for horz
	if (y < 0.5*imHeight)
	{
		update.push_back(2);
		if (y < 0.25*imHeight)
		{
			update.push_back(6);
		}
		else
		{
			update.push_back(7);
		}
	}
	else
	{
		update.push_back(3);
		if (y < 0.75*imHeight)
		{
			update.push_back(8);
		}
		else
		{
			update.push_back(9);
		}
	}


	if (x < 0.5*imWidth)
	{
		update.push_back(4);
		if (x < 0.25*imWidth)
		{
			update.push_back(10);
		}
		else
		{
			update.push_back(11);
		}
	}
	else
	{
		update.push_back(5);
		if (x < 0.75*imWidth)
		{
			update.push_back(12);
		}
		else
		{
			update.push_back(13);
		}
	}

	return 0;
}

//update the features based on location of keypoints 
int HVPartitionRFC::UpdateHVPFeat(int cNum, int x, int y)
{
	vector<int> update_indices;
	GetUpdateIndices(x,y, update_indices);
	for (int i=0; i < update_indices.size(); i++)
	{
		featHVP[update_indices[i]] = featHVP[update_indices[i]] + 1;
	}
	return 0;
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
	pair<float, float> start, end;
	start = partitionBound1[index];
	end = partitionBound2[index];
	float x,y;
	int unidex; 
	float sum_count = 0;
	//cout<<"\n Compute Hist : \n ";
	for( int i = 0; i < imageKeypoints->total; i++ )
    {
        if (cNumVector[i] == -1) // if none of the codeword was assigned
			continue;
		CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( imageKeypoints, i );
        x = r->pt.x;
        y = r->pt.y;
		if (start.first < y &&  y < end.first && start.second < x &&  x < end.second)
		{
			unidex = numCwords*index + cNumVector[i];
			//cout<<cNumVector[i]<< " ";
			featHVP[unidex] = featHVP[unidex] + 1;
			sum_count = sum_count + 1;
		}
        
    }
	//Normalize
	int offset = numCwords*index;
	if (sum_count == 0)
		return 0;
	//cout<<"\n feat vec : \n";
	for (int j=0; j < numCwords ; j++)
	{
		featHVP[offset + j] = featHVP[offset + j]/sum_count;
		//cout<<featHVP[offset + j]<< "  ";
	}
	return 0;
}

int PrintStat(vector<float> vec)
{
	float vmin = *std::min_element(vec.begin(), vec.end());
	float vmax = *std::max_element(vec.begin(), vec.end());
	cout << "\n Min: "<<vmin;
	cout << "\n Max: "<<vmax;
	std::sort(vec.begin(), vec.end());
	int mid_elem=floor(vec.size()/2.0);
	cout<<"\n Median: "<<vec[mid_elem];
	return 0;
}

int HVPartitionRFC::ComputeHVPFeatures()
{
	int i=0;
	int x,y;
	float mindist=0;
	vector<float> mindistDesc;
	featHVP.resize(numPartitions*numCwords);
	//cNumVector.resize( imageKeypoints->total);
	for( i = 0; i < imageKeypoints->total; i++ )
    {
        CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( imageKeypoints, i );
        x = cvRound(r->pt.x);
        y = cvRound(r->pt.y);

        int cNumber = FindNearestCodeWordED(i,mindist); // find in the points
		//cout<<cNumber<<"\t";
		cNumVector.push_back(cNumber);
		mindistDesc.push_back(mindist);
    }
	PrintStat(mindistDesc);
	//UpdateHVPFeat(cNumber, x, y); // histogram features
	ObtainPartition(3,3);// get the boundaries so that features can be extracted 
	//for each partition extract the hist features
	for (i=0; i < partitionBound1.size(); i++)
	{
		ComputeHistFeatures(i);
	}
	return 0;
}



int HVPartitionRFC::ObtainSURFDesc(int size)
{
	descSize = size;
	double tt = (double)cvGetTickCount();
	cv::initModule_nonfree();
	storage = cvCreateMemStorage(0);
	CvSeq* imKeypoints = 0, *imDescriptors = 0;
	CvSURFParams params = cvSURFParams(8000, 0); // 0 for 64 dimensional, 1 for 128 dimensional
	cvExtractSURF( image, 0, &imKeypoints, &imDescriptors, storage, params );
	printf("\nImage Descriptors: %d\n", imDescriptors->total);
    //tt = (double)cvGetTickCount() - tt;
	
	points = cvCreateMat(imDescriptors->total,descSize,CV_32F); // 128 to be removed later
	getDescMat( imKeypoints, imDescriptors, points, 64);
	imageKeypoints = imKeypoints;
	imageDesc = imDescriptors;
	return 0;
}
/*
int HVPartitionRFC::ObtainFREAKDesc(int size)
{
	vector<KeyPoint> keypointsA;
    Mat descriptorsA;
	descSize = size;
	double tt = (double)cvGetTickCount();
	cv::initModule_nonfree();
	storage = cvCreateMemStorage(0);
	CvSeq* imKeypoints = 0, *imDescriptors = 0;
	//CvSURFParams params = cvSURFParams(10000, 0); // 0 for 64 dimensional, 1 for 128 dimensional
	//cvExtractSURF( image, 0, &imKeypoints, &imDescriptors, storage, params );
	SurfFeatureDetector detector(2000,4);
	FREAK extractor;
	detector.detect( image, keypointsA );
	extractor.compute( image, keypointsA, descriptorsA );
    //tt = (double)cvGetTickCount() - tt;
	points = cvCreateMat(imDescriptors->total,descSize,CV_32F); // 128 to be removed later
	getDescMat( imKeypoints, imDescriptors, points, 64);
	imageKeypoints = imKeypoints;
	imageDesc = imDescriptors;
	//printf("Image Descriptors: %d\n", imDescriptors->total);
	return 0;
}*/



vector<float> HVPartitionRFC::getHistFeat()
{
	return featHVP;
}

