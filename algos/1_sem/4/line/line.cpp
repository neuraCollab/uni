#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std;

void countingSort(vector<int>& arr, int n, int exp) {
    vector<int> output(n), count(n, 0);

    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % n]++;
    }

    for (int i = 1; i < n; i++) {
        count[i] += count[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % n] - 1] = arr[i];
        count[(arr[i] / exp) % n]--;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

void radixSort(vector<int>& arr, int n) {
    for (int exp = 1; exp <= n * n; exp *= n) {
        countingSort(arr, n, exp);
    }
}

int main() {
    int n;
    cout << "Введите длину массива: ";
    cin >> n;

    if (n > 1) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, n * n * n - 1);

        vector<int> A(n);
        for (int& value : A) {
            value = dis(gen);
        }

        auto start = chrono::high_resolution_clock::now();
        radixSort(A, n);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> radixSort_time = end - start;

        cout << "Поразрядная сортировка заняла " << radixSort_time.count() << " мс.\n";
    } else {
        cout << "Массив слишком мал для сортировки.\n";
    }

    return 0;
}
