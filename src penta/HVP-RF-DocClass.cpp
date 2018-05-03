#include "HVPartitionRFC.h"
#include "ImageBasedCodeBook.h"
#include "RandForestTrain.h"
#include <stdio.h>
#include<cstring>
#include <iterator>
#include <sstream>

using namespace std;

int	WriteTrainData(const char* filename, const char *trainFname, const char *codebook_filename,unsigned int , unsigned int );
void help();

int main(int argc, char** argv)
{
	int i=0;
	argc = 7;

	/*  argv[1] = "0";
        argv[2] =  "C:\\Users\\chinmay\\Desktop\\image.txt";
        argv[3] = "C:\\Users\\chinmay\\Desktop\\codebook.txt";
        argv[4] = "100";
    */

	/*  argv[1] = "1";
        argv[2] =  "C:\\Users\\chinmay\\Desktop\\image_class2.txt";
        argv[3] = "C:\\Users\\chinmay\\Desktop\\codebook2.txt";
        argv[4] = "C:\\Users\\chinmay\\Desktop\\test_feat_temp2.txt";
        argv[5] = "100";    argv[6] = "64";
    */

	/*  argv[1] = "2";
        argv[2] = "C:\\Users\\chinmay\\Desktop\\test_feat_temp2.txt";
        argv[3] = "1999"; 	argv[4] = "1300";
        argv[5] = "5"; argv[6] = "150";
    */

    /*  argv[1] = "3";
        argv[2] = "C:\\Users\\chinmay\\Desktop\\test_feat_temp2.txt";
        argv[3] = "1998"; 	argv[4] = "1300";
        argv[5] = "5";
        argv[6] = "C:\\Users\\chinmay\\Desktop\\model2.txt";
    */

        argv[1] = "4";
        argv[2] = "C:\\Users\\chinmay\\Desktop\\test2.txt";
        argv[3] = "C:\\Users\\chinmay\\Desktop\\model2.txt";
        argv[4] = "C:\\Users\\chinmay\\Desktop\\codebook2.txt";
        argv[5] = "1300";
//        argv[6] = "100";
        //argv[7] = "64";

    if (argc < 3)
	{
		help();
		exit(0);
	}

	if (strcmp(argv[1] , "0")  == 0)
	{
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
		//cout<<"output begins\n";
		RandForestTrainTest rfT(NumSamp,Numfeat,NumClass); //Number of sample, number of features, number of classes is required
		rfT.read_data(argv[2]);
		if (numTrain >= 1) // number of images per class for training
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
		rfT.read_data(argv[2]); //test_feat_temp.txt
		rfT.TrainTestRF(argv[6]); // save the trained model using given file path
	}
	else if (strcmp(argv[1] , "4")  == 0)
	{
		unsigned int Numfeat = atoi(argv[5]); // number of features
		RandForestTrainTest rfT(Numfeat);
		unsigned int NumCW = 100;//atoi(argv[6]); // number of CW
		unsigned int descsize = 64;//atoi(argv[7]);// desc size 64 or 128
		rfT.RFClassify(argv[2],argv[3],argv[4], NumCW, descsize); // testimage_filename,model_filename,codebook_filename
	}
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
		);
    return;
}


//This is step before training when for each image given in text file we extract features and write in a text file
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
            cout << "class variable = " << classvar << endl;
			IplImage* image = cvLoadImage( tokens[0].c_str(), CV_LOAD_IMAGE_GRAYSCALE );// read image
			if(!image )
			{
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

