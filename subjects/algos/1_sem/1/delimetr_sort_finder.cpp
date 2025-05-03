#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>

using namespace std;

int sum = 0;

// Merges two subarrays of array[].
// First subarray is arr[begin..mid]
// Second subarray is arr[mid+1..end]
void find_sum_pairs(int array[], int const left, int const mid,
           int const right, int const S)
{

    // int const subArrayOne = mid - left + 1;
    int const subArrayTwo = right - mid;


    // Copy data to temp arrays leftArray[] and rightArray[]
    for (int i = left; i < subArrayTwo; i++)
    {
        for (int j = mid; j < right; j++)
        {
            if (array[i] + array[j] == S) sum++;
        }
        
    }
    

}

void merge_Sort(int *array, int start_of_array, int end_of_array, int S)
{
    if ( start_of_array < end_of_array)
    {
        int center = start_of_array + (end_of_array - start_of_array) / 2;

        merge_Sort(array, start_of_array, center, S);
        merge_Sort(array, center + 1, end_of_array, S);

        find_sum_pairs(array, start_of_array, center, end_of_array, S);
    } 
}

int main() 
{

    int N;

    size_t S;

    cin >> N >> S;

    int A[N] = {0};

    for (int i = 1; i <= N; i++)
    {
        A[i - 1] = i;
    }

    merge_Sort(A, 0, N - 1, S);

    cout << (sum ? "Количество пар: " : "Таких пар не существует: ")  << sum << endl;

    return 0;
}