#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

void fillVectorWithRandomNumbers(vector<int> &vec, int size, int minValue, int maxValue)
{
    srand(static_cast<unsigned int>(time(0)));
    vec.clear();

    for (int i = 0; i < size; ++i)
    {
        vec.push_back(rand() % (maxValue - minValue + 1) + minValue);
    }
}

void merge(vector<int> &arr, int from, int medium, int to)
{
    int n1 = medium - from + 1, n2 = to - medium;

    vector<int> left(n1), right(n2);

    for (int i = 0; i < n1; i++)
        left[i] = arr[from + i];
    for (int j = 0; j < n2; j++)
        right[j] = arr[medium + j + 1];

    int i = 0, j = 0, k = from;
    while (i < n1 && j < n2)
    {
        if (left[i] <= right[j])
        {
            arr[k] = left[i];
            i++;
        }
        else
        {
            arr[k] = right[j];
            j++;
        }

        k++;
    }

    while (i < n1)
    {
        arr[k] = left[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        arr[k] = right[j];
        j++;
        k++;
    }
}

void mergeSort(vector<int> &arr, int from_sort, int to_sort)
{
    if (from_sort < to_sort)
    {
        int medium = from_sort + (to_sort - from_sort) / 2;

        mergeSort(arr, from_sort, medium);
        mergeSort(arr, medium + 1, to_sort);

        merge(arr, from_sort, medium, to_sort);
    }
}

int main()
{
    int n;
    cin >> n;

    vector<int> arr_for_sort;

    if (n > 0)
        fillVectorWithRandomNumbers(arr_for_sort, n, 1, 100);

    
    // for (int num : arr_for_sort)
    // {
    //     cout << num << " ";
    // }

    mergeSort(arr_for_sort, 0, arr_for_sort.size() - 1);

    cout << "Отсортированный массив: ";
    for (int num : arr_for_sort)
    {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}
