// Random Forest wrapper class for training, prediction and cross
// validation.
// Author: Jayant Kumar, jayant@umiacs.umd.edu

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file
#ifndef RAND_FOREST_TRAIN
	#include "RandForestTrain.h"
#define RAND_FOREST_TRAIN
#endif
#ifndef HVP_PARTITION
	#include "HVPartitionRFC.h"
#define HVP_PARTITION
#endif

using namespace std;
using namespace cv; // OpenCV API is in the C++ "cv" namespace

//Set Rfmodel to classify new images
int RandForestTrainTest::SetRFModel(const char *model_file)
{
	rtree = new CvRTrees;
	rtree->load(model_file);
	return 0;
}

//obtain count of each class, corresponding indices for CV
int RandForestTrainTest::GetClassStat()
{
	int count;
	for (int class_iter = 1; class_iter <= numClasses ; class_iter++)
	{
		count = 0;
		vector<int> cl_indx;
		for (int n=0 ; n < numTrainSample; n++)
		{
			if (data_class.at<float>(n,0) == class_iter)
			{
				count++;
				cl_indx.push_back(n);
			}
		}
		class_count.push_back(count);
		class_index.push_back(cl_indx);
	}
	return 0;
}

//Feat matrix has been set, model has been loaded, now predict, confidence TBD
int	RandForestTrainTest::GetRFClass(double &result,double &confidence)
{
	Mat test_sample = featMat.row(0);
	result = rtree->predict(test_sample, Mat());
	//result = rtree->predict_prob(test_sample, Mat());
	
	return 0;
}

//Given a list of image file paths, and a model file name, and codebook path
//predict the class using the trained model
int	RandForestTrainTest::RFClassify(const char* filename, const char *model_filename, const char *codebook_filename,
	unsigned int numCW, unsigned int descsize)
{
	int i;
	double doc_class, doc_conf;
	string imagefilename;
	std::ifstream imageFList (filename);
	featMat = Mat(1, numFeat, CV_32FC1);
	SetRFModel(model_filename);
	if (imageFList.is_open() )
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
			//float classvar = atof(tokens[1].c_str()); // Ignore class labels

			IplImage* image = cvLoadImage( tokens[0].c_str(), CV_LOAD_IMAGE_GRAYSCALE );// read image
			if(!image )
			{
				//fprintf( stderr, "Can not load %s\n",scene_filename );
				std::cout << "Error opening imagefile : " << imagefilename << std::endl;
				return -1;
			}
			HVPartitionRFC hvpObj(image,3,3);
			hvpObj.read_codebook(codebook_filename, numCW, descsize);
			hvpObj.ObtainSURFDesc(descsize);
			hvpObj.ComputeHVPFeatures();
			vector<float> histFeat = hvpObj.getHistFeat();
			for (i=0; i < histFeat.size(); i++) // space separated features in each row
				featMat.at<float>(0,i) = histFeat[i];
			//obtain class
			GetRFClass(doc_class, doc_conf);
			cout<<"\n Class : "<<doc_class<<"\n";
		}
		imageFList.close();
	}
	else
	{
		std::cout << "Unable to open file" << std::endl << std::endl;
		return -1;
	}
	return 0;
}


// loads the sample database from file 
int RandForestTrainTest::read_data(const char* filename)
{
    float tmpVal;
	int i=0, j=0;
    // if we can't read the input file then return 0
    ifstream datafile (filename);
	featMat = Mat(numTrainSample, numFeat, CV_32FC1);
	data_class = Mat(numTrainSample, 1, CV_32FC1);
	if (datafile.is_open())
	{
		while (datafile >> tmpVal)
		{
			featMat.at<float>(i,j) = tmpVal;
			j++;
			if (j == numFeat)
			{
				datafile >> tmpVal; // last attribute
				data_class.at<float>(i,0) = tmpVal;
				j=0;
				i++;
			}
		}
		
	}
	else
	{
		std::cout << "Unable to open file" << std::endl;
		return -1;
	}
    
	datafile.close();
    return 0;
}

// Create aux. data for train RF to obtain proximities
int RandForestTrainTest::create_aux_data()
{
	int v1,i,j;
	//auxFeatMat = Mat(numTrainSample, numFeat, CV_32FC1);
	int nTS = numTrainSample/2; // must be an integer
	int step = nTS;
	for(i=0; i < nTS; i++)
	{
		for (j=0; j < numFeat; j++)
		{
			// generate a random number in the 0 - numTrainSample-1
			int riter = rand() % nTS; // numTrainSample < 32567 ? otherwise use other function
			featMat.at<float>(i + nTS,j) = featMat.at<float>(riter,j);
		}
		data_class.at<float>(i + nTS,0) = 2; // aux. data second class
		data_class.at<float>(i,0) = 1; // orig. data is Class 1, overwite what has been read from file
	}

	return 0;
}

// This module trains a RF for obtaining proximity between data points for clustering
int RandForestTrainTest::ObtainProximity(const char *prox_fname)
{
	double acc = 0;
	std::ofstream prox_file (prox_fname);
	if (!prox_file.is_open())
	{
		cout << "Error opening file : " << prox_fname << std::endl;
		return -1;
	}
	float priors[] = {1,1}; // only two classes - aux. and orig
	params = CvRTParams(33, // max depth
                                4, // min sample count
                                0, // regression accuracy: N/A here
                                false, // compute surrogate split, no missing data
                                10, // max number of categories (use sub-optimal algorithm for larger numbers)
                                priors, // the array of priors
                                false,  // calculate variable importance
                                0,       // number of variables randomly selected at node and used to find the best split(s).
                                100,	 // max number of trees in the forest
                                0.01f,				// forrest accuracy
                                CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
                                );


	Mat var_type = Mat(numFeat+1, 1, CV_8U ); // 1 for class category
    
	var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
    var_type.at<uchar>(numFeat, 0) = CV_VAR_CATEGORICAL; // last one for class categorical

    CvRTrees* rtree = new CvRTrees;
	rtree->train(featMat, row_sample, data_class, Mat(), Mat(), var_type, Mat(), params);
	
	// now do a O(N^2) iterations for computing proximity
    int i,j;
	int nTS = numTrainSample/2; // now back to orig data
	CvMat sample1, sample2;
	rfProximity =    Mat(nTS, nTS, CV_32FC1);  
	for (i=0; i < nTS; i++)
	{
		sample1  = featMat.row(i);
		rfProximity.at<float> (i, i) = 1;
		for (j=i+1 ; j < nTS; j++)
		{
			sample2  = featMat.row(j);
			rfProximity.at<float> (i, j) =  rtree->get_proximity(&sample1, &sample2);
			rfProximity.at<float> (j, i) = rfProximity.at<float> (i, j); // symmetric
			//cout << "\n proximity : "<< rfProximity.at<float> (i, j);
		}
	}
	for (i=0; i < nTS; i++)
	{
		for (j=0; j < nTS; j++)
			prox_file << rfProximity.at<float> (i, j) << "  ";
		prox_file << "\n";
	}

	prox_file.close();
	return 0;
}


/** Cross-validation with percentage of training samples specified (per class) **/
int RandForestTrainTest::do_CrossValidation(double percentTrain)
{
	std::srand ( unsigned ( std::time(0) ) );
	int numIter = 10;
	int numSampTrain;
	GetClassStat(); // get the count of each class

	RNG rng(12345);
    for (int iter = 0; iter < numIter; iter++)
    {
		//std::random_shuffle ( points.begin(), points.end() );   
        // run random forest prediction
		vector<int> trainpoints;
		vector<int> testpoints;
		for (int k =0; k < numClasses; k++)
		{
			std::vector<int> points;
			vector<int> cls_index = class_index[k];
			if (percentTrain < 1) // if the number of sample is determeined by a percentage
				numSampTrain = floor(percentTrain*cls_index.size());
			else
				numSampTrain = percentTrain;
			
			if (cls_index.size() < numSampTrain) // What if there is not enough elements 
			{
				numSampTrain = cls_index.size(); // use the whole for training no testing, only happens when fixed no. of train samples given  
			}

			for (int i=0; i < cls_index.size(); ++i) 
				points.push_back(cls_index[i]);
			//shuffle
			std::random_shuffle ( points.begin(), points.end() ); 
			//train
			for (int p=0; p < numSampTrain; p++)
				trainpoints.push_back( points.at(p));
			//test 
			for (int p=numSampTrain; p < cls_index.size(); p++)
				testpoints.push_back( points.at(p));
			
		
		}
		accuracy.push_back( Cross_validate(trainpoints,testpoints));
	}
	sort(accuracy.begin(),accuracy.end());
	cout<<"\n Median acuracy : "<< accuracy.at(numIter/2);
        
	return 0;

}


// This module performs one iteration of cross validation
double RandForestTrainTest::TrainTestRF(const char * model_filename, bool trainflag)
{
	double acc = 0;

	float *priors= new float[numClasses] ;//= {1,1,1,1};
	for (int i =0; i< numClasses; i++)
		priors[i] = 1;
	params = CvRTParams(20, // max depth
                                2, // min sample count
                                0, // regression accuracy: N/A here
                                false, // compute surrogate split, no missing data
                                numClasses, // max number of categories (use sub-optimal algorithm for larger numbers)
                                priors, // the array of priors
                                true,  // calculate variable importance
                                0,       // number of variables randomly selected at node and used to find the best split(s).
                                500,	 // max number of trees in the forest
                                0.01f,				// forrest accuracy
                                CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
                                );


	Mat var_type = Mat(numFeat+1, 1, CV_8U ); // 1 for class category
    var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

    // this is a classification problem (i.e. predict a discrete number of class
    // outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL
    var_type.at<uchar>(numFeat, 0) = CV_VAR_CATEGORICAL;
	// train random forest classifier (using training data)

    CvRTrees* rtree = new CvRTrees;
	rtree->train(featMat, row_sample, data_class, Mat(), Mat(), var_type, Mat(), params);
	rtree->save(model_filename);
	//display some stats related to training
	//cout<<"\n The training error is "<<rtree->get_train_error();
	rtree->load(model_filename);
	//cout<<"\n\n Displaying var importance: \n";
	//Mat var_imp = rtree->getVarImportance();
	//for (int j=0; j < numFeat; j++)
	//	cout<<var_imp.at<float>(j,0)<<"  ";
	//do some test
	CvRTrees* rtree_test = new CvRTrees;
	rtree_test->load(model_filename);
	Mat test_sample;
	double result;
	for (int tsample = 0; tsample < numTrainSample; tsample++)
    {
            // extract a row from the testing matrix
            test_sample = featMat.row(tsample);
			result = rtree_test->predict(test_sample, Mat());
			cout<<"class : "<<result<<endl;
	}
	return 0;
}

// This module performs one iteration of cross validation given a vector of train and test indices
double RandForestTrainTest::Cross_validate(vector<int> trainIndex, vector<int> testIndex)
{
	double acc = 0;
	Mat training_data = Mat(trainIndex.size(),numFeat, CV_32FC1);
    Mat training_classifications = Mat(trainIndex.size(), 1, CV_32FC1);
	
	Mat testing_data = Mat(testIndex.size(), numFeat, CV_32FC1);
    Mat testing_classifications = Mat(testIndex.size(), 1, CV_32FC1);

	//Copy train data to a new array
	for (unsigned int row=0;row < trainIndex.size(); row++)
	{
		for(int attribute = 0; attribute < numFeat ; attribute++)
		  {
			  training_data.at<float>(row,attribute) = featMat.at<float>(trainIndex.at(row),attribute);
		  }
		  training_classifications.at<float>(row,0) = data_class.at<float>(trainIndex.at(row),0);
	}
	for (unsigned int row=0;row < testIndex.size(); row++)
	{
		  for(int attribute = 0; attribute < numFeat ; attribute++)
		  {
			  testing_data.at<float>(row,attribute) = featMat.at<float>(testIndex.at(row),attribute);
		  }
		  testing_classifications.at<float>(row,0) = data_class.at<float>(testIndex.at(row),0);
	}

	float *priors= new float[numClasses] ;//= {1,1,1,1};
	for (int i =0; i< numClasses; i++)
		priors[i] = 1;
	params = CvRTParams(33, // max depth
                                2, // min sample count
                                0, // regression accuracy: N/A here
                                false, // compute surrogate split, no missing data
                                80, // max number of categories (use sub-optimal algorithm for larger numbers)
                                priors, // the array of priors
                                true,  // calculate variable importance
                                0,       // number of variables randomly selected at node and used to find the best split(s).
                                100,	 // max number of trees in the forest
                                0.01f,				// forrest accuracy
                                CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
                                );


	Mat var_type = Mat(numFeat + 1, 1, CV_8U );
    var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
    var_type.at<uchar>(numFeat, 0) = CV_VAR_CATEGORICAL;

    CvRTrees* rtree = new CvRTrees;

    rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,Mat(), Mat(), var_type, Mat(), params);
	Mat test_sample;
    int correct_class = 0;
    int wrong_class = 0;
	double result;
 
    for (int tsample = 0; tsample < testIndex.size(); tsample++)
    {
        // extract a row from the testing matrix
        test_sample = testing_data.row(tsample);

        // run random forest prediction
        result = rtree->predict(test_sample, Mat());

        if (fabs(result - testing_classifications.at<float>(tsample, 0)) >= FLT_EPSILON)
        {
            // if they differ more than floating point error => wrong class
            wrong_class++;
        }
        else
        {
            correct_class++;
        }
    }
	
	acc = (double) correct_class*100.0/testIndex.size();
	cout<<"\n accuracy: "<<acc;
	return acc;

}
