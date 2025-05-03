#include<iostream>

using namespace std;

const int N = 1001;
int a[N] {};


int main()
{
    int n;

    cout << "Длинна массива: ";
    cin >> n;

    for(int i = 0; i < n; i++)
    {
        cin >> a[i];
    }

    for(int i = 0; i < n - 1; i++)
    {
        if(i > 1 && i % 2 != 0)
        {
            if (a[i] >= a[i - 1])
            {
                swap(a[i-1], a[i]);
            }
        } else if(i % 2 == 0) {
            
            if(a[i] <= a[i - 1])
            {
                swap(a[i], a[i-1]);
            }
        }
    }

    for ( int i = 0; i < n; i ++) {
        cout << a[i] << "  " ;
    }

}