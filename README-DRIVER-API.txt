The main module is in HVP-RF-DocClass.cpp
	
        "Usage: 
		./HVP-RF-UMDDocClass 0 image_filename codebook_filename NumCodewords => For codebook creation 
		./HVP-RF-UMDDocClass 1 image_filename codebook_filename train_feat_filename NumCodewords Descsize => For writing features to a text file (for training), last feature is class label) 
		./HVP-RF-UMDDocClass 2 train_feat_filename NumSamples NumFeatures NumClasses NumTrain=> For cross validation results 
		./HVP-RF-UMDDocClass 3 train_feat_filename NumSamples NumFeatures NumClasses model_filename => For training RF and saving model 
		./HVP-RF-UMDDocClass 4 testimage_filename model_filename codebook_filename NumFeatures NumCW Descsize => For classification of images
		 
		
		
		where 
		for option 0:
		image_filename is the file containing path to images in each line. These images are to be used for codebook creation.
		codebook_filename is the file in which created codebook will be saved for late use.
		NumCodewords is the given number of codewords to be learnt (usually set 100 or 200)	
	    
		For option 1:
		image_filename is the file containing path to training images in each line. In the second column of each row mention the class(number).
		example:
		C:\Users\jayant\Xerox_work\Data\doc1.tif	1
		C:\Users\jayant\Xerox_work\Data\doc2.tif	2
		
		Features will be extracted from these images.
		
		codebook_filename is the codebook file name learned fom previous step. option 0
		
		train_feat_filename is the file in which features will be written
		
		NumCodewords = number of codewords
		Descsize = 64
		
		For option 2,3,4:
		First perform option 1 (i.e. extract features and write to a file)	
		NumSamples = number of image samples
		NumFeatures = number of features (13 * no. of codewords, 1300 if NumCodeword = 100)
		model_filename = file in which RF model will be saved
		
		NumTrain = Number of training samples per class (set 1)
		
		For option 4:
		testimage_filename = file contains path to images for which label needs to be predicted
		Example: (No class label at the end)
		C:\Users\jayant\Xerox_work\Data\doc1.tif
		model_filename = file in which RF model will be saved (using option 3)
		NumCodewords = number of codewords (in option 0)
		Descsize = 64
		
		
		
To link opencv with codeblocks visit this page:
	https://kevinhughes.ca/tutorials/opencv-install-on-windows-with-codeblocks-and-mingw 
		
