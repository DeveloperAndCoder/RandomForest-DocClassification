#include <dirent.h>
#include <string>
#include <vector>
#include<fstream>
#include<sstream>
#include<iostream>

using namespace std;

int main() {
	/*string path = "C:\\Users\\chinmay\\Desktop\\Table_Dataset\\Table_Dataset\\RELEVANT\\";
	 DIR*    dir;
    dirent* pdir;
    std::vector<std::string> files;
	
	dir = opendir(path.c_str());
	ofstream out("C:\\Users\\chinmay\\Desktop\\image_class2.txt");
	int sum = 0;
	int i = 1;
	bool t = false;
	while ((pdir = readdir (dir)) != NULL) {
    	printf ("%s\n", pdir->d_name);
    	if(i == 12)
    		t = true;
    	if(t){
    		out<< "C:\\Users\\chinmay\\Desktop\\Table_Dataset\\Table_Dataset\\RELEVANT\\"<< pdir->d_name <<'\t'<<"1\n";
    		sum++;
		}
    	i++;
//    	if(i == (149 + 251))
//    		break;
    }
  	closedir (dir);
  	
	i = 1;
  	path = "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\ADVE\\";
  	dir = opendir(path.c_str());
  	while ((pdir = readdir (dir)) != NULL) {
    	printf ("%s\n", pdir->d_name);
if(i == 12)
    		t = true;
    	if(t){    		out<< "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\ADVE\\"<< pdir->d_name <<'\t'<<"2\n";
    		sum++;
		}
    	i++;
    }
  	closedir (dir);
    
    i = 1;
  	path = "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Email\\";
  	dir = opendir(path.c_str());
  	while ((pdir = readdir (dir)) != NULL) {
    	printf ("%s\n", pdir->d_name);
if(i == 12)
    		t = true;
    	if(t){    		out<< "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Email\\"<< pdir->d_name <<"\t3\n";
    		sum++;
		}
    	i++;
    }
  	closedir (dir);
  	
  	i = 1;
  	path = "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Form\\";
  	dir = opendir(path.c_str());
  	while ((pdir = readdir (dir)) != NULL) {
    	printf ("%s\n", pdir->d_name);
if(i == 12)
    		t = true;
    	if(t){    		out<< "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Form\\"<< pdir->d_name <<"\t4\n";
    		sum++;
		}
    	i++;
    }
  	closedir (dir);
  	
  	i = 1;
  	path = "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Letter\\";
  	dir = opendir(path.c_str());
  	while ((pdir = readdir (dir)) != NULL) {
    	printf ("%s\n", pdir->d_name);
if(i == 12)
    		t = true;
    	if(t){    		out<< "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Letter\\"<< pdir->d_name <<"\t5\n";
    		sum++;
		}
    	i++;
    }
  	closedir (dir);
  	/*
  	i = 1;
  	path = "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Email\\";
  	dir = opendir(path.c_str());
  	while ((pdir = readdir (dir)) != NULL) {
    	printf ("%s\n", pdir->d_name);
    	if(i%23 == 0 && i%12 != 0){
    		out<< "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Email\\"<< pdir->d_name <<'\n';
    		sum++;
		}
    	i++;
    }
  	closedir (dir);
  	
  	i = 1;
  	path = "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Letter\\";
  	dir = opendir(path.c_str());
  	while ((pdir = readdir (dir)) != NULL) {
    	printf ("%s\n", pdir->d_name);
    	if(i%23 == 0 && i%12 != 0){
    		out<< "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Letter\\"<< pdir->d_name <<'\n';
    		sum++;
		}
    	i++;
    }
  	closedir (dir);
	
	i = 1;
  	path = "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\ADVE\\";
  	dir = opendir(path.c_str());
  	while ((pdir = readdir (dir)) != NULL) {
    	printf ("%s\n", pdir->d_name);
    	if(i%9 == 0 && i%4 != 0){
    		out<< "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\ADVE\\"<< pdir->d_name <<'\n';
    		sum++;
		}
    	i++;
    }
  	closedir (dir);
  	
  	i = 1;
  	path = "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Form\\";
  	dir = opendir(path.c_str());
  	while ((pdir = readdir (dir)) != NULL) {
    	printf ("%s\n", pdir->d_name);
    	if(i%15 == 0 && i%8 != 0){
    		out<< "C:\\Users\\chinmay\\Desktop\\Tobacco3482_2\\Form\\"<< pdir->d_name <<'\n';
    		sum++;
		}
    	i++;
    }
  	closedir (dir);
	*/
	/*
	cout<<endl<<sum<<endl;
*/

cout << "\033[1;31mbold red text\033[0m\n";  
cout << "\033[" << "yo" << "m";  
return 0;
}
