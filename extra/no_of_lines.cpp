#include <iostream>
#include <fstream>
using namespace std;

int main(){
	ifstream aFile ("test.txt");
if (aFile.good()) {
//how do i get total file line number?
char c;
int i = 0;
while (aFile.get(c))
    if (c == '\n')
        ++i;
cout<<i<<endl;
}
}

