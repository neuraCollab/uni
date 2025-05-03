#include <iostream>
#include  <cmath>
#include  <random>
#include <vector>
#include <algorithm>

using namespace std;


void insertionsort(std::vector<int>::iterator l, std::vector<int>::iterator r) {
	for (std::vector<int>::iterator i = l + 1; i < r; i++) {
		std::vector<int>::iterator j = i;
		while (j > l && *(j - 1) > *j) {
			swap(*(j - 1), *j);
			j--;
		}
	}
}

void shellsortpratt(int* l, int* r, std::vector<int>::iterator steps) {
	int sz = r - l;
	steps[0] = 1;
	int cur = 1, q = 1;
	for (int i = 1; i < sz; i++) {
		int cur = 1 << i;
		if (cur > sz / 2) break;
		for (int j = 1; j < sz; j++) {
			cur *= 3;
			if (cur > sz / 2) break;
			steps[q++] = cur;
		}
	}
	insertionsort(steps, steps + q);
	q--;
	for (; q >= 0; q--) {
		int step = steps[q];
		for (int *i = l + step; i < r; i++) {
			int *j = i;
			int *diff = j - step;
			while (diff >= l && *diff > *j) {
				swap(*diff, *j);
				j = diff;
				diff = j - step;
			}
		}
	}
}

int main () 
{
    int N;
    double time_spent = 0.0, time_spent2 = 0.0;

    cout << "Введите размер массива: ";
    cin >> N;

    int arr[N] = {0};
	vector <int> pratt_arr = {1};
    int* last_index = &arr[N - 1];

    mt19937 mt ( time ( nullptr ) ) ;
    for ( int i = 0; i < N ; ++ i )
        arr [i] = ( mt () % 100) + 100;

    
 
    clock_t begin = clock();
    shellsortpratt(arr, last_index, pratt_arr.begin());    
    clock_t end = clock();


    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    


    clock_t begin2 = clock();
    std::sort(arr, last_index);
    clock_t end2 = clock();


    time_spent2 += (double)(end2 - begin2) / CLOCKS_PER_SEC;
    
    printf("The elapsed time for shell sort is %f seconds, for SORT is %f seconds \n", time_spent, time_spent2);

    return 0;
}