// Implementation of Horizontal-Vertical partitioning class for structural similarity based document classification
// Main program for selecting from different alternatives: codebook creation, training, cross-validation, proximity computation
// Author: Jayant Kumar, jayant@umiacs.umd.edu

#include "HVPartitionRFC.h"
#include "ImageBasedCodeBook.h"
#include "RandForestTrain.h"
#include <stdio.h>
#include <string.h>
#include <iterator>
#include <sstream>

using namespace std;

int	WriteTrainData(const char* filename, const char *trainFname, const char *codebook_filename,unsigned int , unsigned int );
void help();

int main(int argc, char** argv)
{

	int i=0;
	//argc = 7;
	//argv[1] = "2";
	//const char* image_filename =  "C:\\Users\\jayant\\LAMP\\grouping\\TableImageInCLass.txt";
	//argv[2] =  "C:\\Users\\USX26430\\Xerox_work\\CPPcode\\DocClass\\zip\\jfeat.txt"; // for codebook 
	//argv[3] = "\\TIDCodeBook.txt";
	//const char* testimage_filename = "C:\\Users\\jayant\\LAMP\\grouping\\TableImageTest.txt";
	//argv[2] = "C:\\Users\\USX26430\\Xerox_work\\Data\\AlignBasedResults\\TIDImageHVPFeatures.txt";
	//argv[3] = "6"; argv[4] = "1300"; argv[5] = "2"; argv[6] = "1";
 	//argv[3] = "C:\\Users\\USX26430\\Xerox_work\\Data\\AlignBasedResults\\RFmodelTID.txt";
	//argv[5] = "2600";
	//const char* prox_filename = "C:\\Users\\jayant\\LAMP\\grouping\\ProximityTableData.txt";
	//ImageBasedCodeBook iBC;
	//iBC.CreateCodeBook(image_filename, codebook_filename, 100);
	//argc = 6;
	
	if (argc < 3)
	{
		help();
		exit(0);
	}

	if (strcmp(argv[1] , "0")  == 0)
	{
		//CreateCodeBook(argv[2], argv[3]);
		ImageBasedCodeBook iBC;
		unsigned int NumCW = atoi(argv[4]); // number of codewords
		iBC.CreateCodeBook(argv[2], argv[3], NumCW);
	}
	else if (strcmp(argv[1] , "1")  == 0)
	{
		unsigned int NumCW = atoi(argv[5]); // number of CW
		unsigned int descsize = atoi(argv[6]);// desc size 64 or 128
		WriteTrainData(argv[2], argv[4], argv[3], NumCW, descsize); //for writing features, 3 arguments required
	}
	else if (strcmp(argv[1] , "2")  == 0)
	{
		unsigned int NumClass = atoi(argv[5]);
		if (NumClass < 2)
		{
			std::cout << "Invalid no. of classes" << std::endl;
			help();
			exit(0);
		}
		unsigned int NumSamp = atoi(argv[3]);
		unsigned int Numfeat = atoi(argv[4]);
		unsigned int numTrain = atoi(argv[6]);
		RandForestTrainTest rfT(NumSamp,Numfeat,NumClass); //Number of sample, number of features, number of classes is required 
		rfT.read_data(argv[2]);
		if (numTrain >= 1) // numer of images per class for training
			rfT.do_CrossValidation(numTrain);
		else
		{
			std::cout << "Invalid no. of training samples" << std::endl;
			help();
			exit(0);
		}
	}
	else if (strcmp(argv[1] , "3")  == 0)
	{
		unsigned int NumClass = atoi(argv[5]);
		unsigned int NumSamp = atoi(argv[3]);
		unsigned int Numfeat = atoi(argv[4]);
		RandForestTrainTest rfT(NumSamp,Numfeat, NumClass); //Number of sample, number of features, number of classes is required 
		rfT.read_data(argv[2]);
		rfT.TrainTestRF(argv[6]); // save the trained model using given file path
	}
	else if (strcmp(argv[1] , "4")  == 0)
	{
		unsigned int Numfeat = atoi(argv[5]); // number of features
		RandForestTrainTest rfT(Numfeat);
		unsigned int NumCW = atoi(argv[6]); // number of CW
		unsigned int descsize = atoi(argv[7]);// desc size 64 or 128
		rfT.RFClassify(argv[2],argv[3],argv[4], NumCW, descsize); // testimage_filename,model_filename,codebook_filename
	}
	/*else
	{
		unsigned int NumClass = atoi(argv[5]);
		unsigned int NumSamp = atoi(argv[3]);
		unsigned int Numfeat = atoi(argv[4]);
		RandForestTrainTest rfT(2*117,1300, 2); // Use twice the number of samples, No. of classes = 2
		rfT.read_data(train_filename); // features have already been written but ignote the classes
		rfT.create_aux_data(); // this fill up the aux data in feature matrix
		rfT.ObtainProximity(); // train RF and obtain N x N proximity
	}*/

	//WriteTrainData(image_filename, train_filename, codebook_filename); //for writing features
    
	/*** Training RF using features written in a text file, each row contains 1 sample, columns in features, last column is class ****/

	//RandForestTrainTest rfT(824,1300, 53); //Number of sample, number of features, number of classes is required 
	//rfT.read_data(train_filename);
	//rfT.do_CrossValidation(5);   // only cross validation for estimating accuracies 
	//rfT.TrainTestRF(model_filename); // save the trained model using given file path
	
	/** Classify new images using saved model (model_filename), and saved codebook  **/
	//RandForestTrainTest rfT(1300);
	//rfT.RFClassify(testimage_filename,model_filename,codebook_filename);
	
	/** Create Auxilliary data, train RF, and obtain proximity for clustering  **/
	/*RandForestTrainTest rfT(824,1300, 2, true); // Use twice the number of samples, No. of classes does not matter (give 2)
	rfT.read_data(train_filename); // features have already been written but ignore the classes
	rfT.create_aux_data(); // this fill up the aux data in feature matrix
	rfT.ObtainProximity(prox_filename); // train RF and obtain N x N proximity, write the matrix to a file and run suitable scripts for clustering
	*/
	return 0;
}

void help()
{
    printf(
        "This program performs structural similarity based document classification and grouping.\n"
        "Usage: \n"
		"./HVP-RF-UMDDocClass 0 image_filename codebook_filename NumCodewords => For codebook creation \n"
		"./HVP-RF-UMDDocClass 1 image_filename codebook_filename train_feat_filename NumCodewords Descsize=> For writing features to a text file (for training), last feature is class label) \n"
		"./HVP-RF-UMDDocClass 2 train_feat_filename NumSamples NumFeatures NumClasses NumTrain=> For cross validation results \n"
		"./HVP-RF-UMDDocClass 3 train_feat_filename NumSamples NumFeatures NumClasses model_filename => For training RF and saving model \n"
		"./HVP-RF-UMDDocClass 4 testimage_filename model_filename codebook_filename NumFeatures NumCW Descsize => For classification of images\n"
		"./HVP-RF-UMDDocClass 5 train_feat_filename proximity_filename => For creating aux. data, training RF for proximity, and writing proximity \n"
		 );
    return;
}


//This is step before traning when for each image given in text file we extract features and write in a text file 
//along with class (last column)
int	WriteTrainData(const char* filename, const char *trainFname, const char *codebook_filename, unsigned int numCW, 
	unsigned int descSize)
{
	int i;
	string imagefilename;
	std::ifstream imageFList (filename);
	std::ofstream trainFile (trainFname);

	if (imageFList.is_open() && trainFile.is_open())
	{
		while (imageFList.good())
		{
			getline(imageFList,imagefilename); //read image path
			if (imagefilename.empty())
				break;
			//split image name and class
			std::vector<std::string> tokens;
			istringstream iss(imagefilename);
			copy(istream_iterator<string>(iss),istream_iterator<string>(),back_inserter<vector<string> >(tokens));	
			
			std::cout << tokens[0] << std::endl;
			float classvar = atof(tokens[1].c_str());

			IplImage* image = cvLoadImage( tokens[0].c_str(), CV_LOAD_IMAGE_GRAYSCALE );// read image
			if(!image )
			{
				//fprintf( stderr, "Can not load %s\n",scene_filename );
				std::cout << "Error opening imagefile : " << imagefilename << std::endl;
				return -1;
			}
			HVPartitionRFC hvpObj(image,3,3); // 3 levels of Horz. and Vert. partition
			hvpObj.read_codebook(codebook_filename, numCW, descSize);
			hvpObj.ObtainSURFDesc(descSize);
			hvpObj.ComputeHVPFeatures();
			vector<float> histFeat = hvpObj.getHistFeat();
			for (i=0; i < histFeat.size(); i++) // space separated features in each row
				 trainFile <<  histFeat[i] << " ";
			//write class at the end
			trainFile << classvar;
			trainFile << "\n"; // change line for next feature
			cvReleaseImage(&image);
		}
		imageFList.close();
		trainFile.close();
	}
	else
	{
		std::cout << "Unable to open file" << std::endl << std::endl;
		return -1;
	}
	return 0;
}