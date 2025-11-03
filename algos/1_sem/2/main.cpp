#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

using namespace std;
int median(vector<int> &v)
{
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}
void getApprox(int x, int y,  double a, double b, int n) {
  int sumx = 0;
  int sumy = 0;
  int sumx2 = 0;
  int sumxy = 0;
  for (int i = 0; i<n; i++) {
    sumx += x;
    sumy += y;
    sumx2 += x * x;
    sumxy += x * y;
  }
  a = (n*sumxy - (sumx*sumy)) / (n*sumx2 - sumx*sumx);
  b = (sumy - a*sumx) / n;
}

int main () {
    int n = 6;
    string a = "5.txt";
    while (--n != 0)
    {
    a[0] = to_string(n)[0];
    cout << endl<< a << endl;
ifstream in(a);
if (!in.is_open())
    return 0;

std::ifstream inFile(a);
int NumOfTests = std::count(std::istreambuf_iterator<char>(inFile),
                            std::istreambuf_iterator<char>(), '\n') + 1;
vector <int> MeansInt (NumOfTests);
vector<int> Differences (NumOfTests);
int nCounter = 0;
while (nCounter++ < NumOfTests){
    int SumOfSpeeds = 0;
    int AmountOfThreads = 10;
    while (AmountOfThreads-- > 0){
        int temp = 0;
    in >> temp;
    //cout << temp << " ";
    SumOfSpeeds += temp;
    }
    AmountOfThreads = 10;
    while (AmountOfThreads-- > 0){
        int temp = 0;
        in >> temp;
//    cout << temp << " ";
    }
    int Total = 0;
    in >> Total;
    //cout << "ACT VALUE: " << Total << endl;
    Total *= 125000;
    double Mean = Total / double(SumOfSpeeds);
    Total /= 125000;
    SumOfSpeeds *= 10.4139;
    SumOfSpeeds /= 125000;
    int Difference = abs(Total - SumOfSpeeds);
    Differences[nCounter] = Difference;
//    cout << "HELLO HERE IS YOUR MEAN == " << Mean << endl;
//    cout <<"MY VALUE: " << SumOfSpeeds << endl;
//    cout << "HELLO HERE IS YOUR DIFF == " << Difference << endl;
    MeansInt[nCounter] = Mean * 10000;
}
in.close();
sort(Differences.begin(), Differences.end());
sort(MeansInt.begin(), MeansInt.end());
cout <<"Median Difference: " <<  median(Differences) << endl;
cout <<"Correctness: "  << 100. / (median (Differences) + 1) << "%" << endl;
cout <<"Median Quotient: " << median(MeansInt)/10000.;
    }
//double Sum = 0;
//for (double x : Means)
//    Sum += x;
//cout << Sum / NumOfTests;
return 0;
}
