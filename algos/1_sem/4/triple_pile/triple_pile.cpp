#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

void heapify(vector<int>& A, int n, int i) {
    int largest = i;
    int left = 3*i + 1;
    int middle = 3*i + 2;
    int right = 3*i + 3;

    if (left < n && A[left] > A[largest])
        largest = left;

    if (middle < n && A[middle] > A[largest])
        largest = middle;

    if (right < n && A[right] > A[largest])
        largest = right;

    if (largest != i) {
        swap(A[i], A[largest]);
        heapify(A, n, largest);
    }
}

void ternaryHeapSort(vector<int>& A) {
    int n = A.size();
    // Создаем очень большую кучу чего то, я бы сказал даже максимальную!
    for (int i = n / 3 - 1; i >= 0; i--)
        heapify(A, n, i);

    // сортируем это г... Кучу!
    for (int i = n - 1; i > 0; i--) {
        swap(A[0], A[i]);
        heapify(A, i, 0);
    }
}

int main() {
    int n;
    cout << "Введите длину массива: ";
    cin >> n;


    // Взял отсюда:
    // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100);

    vector<int> A(n);
    for (int& value : A) {
        value = dis(gen);
    }

    auto start = chrono::high_resolution_clock::now();
    ternaryHeapSort(A);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ternaryHeapSort_time = end - start;
    cout << "Тернарная пирамидальная сортировка заняла " << ternaryHeapSort_time.count() << " мс.\n";

    // Для сравнения с std::sort
    vector<int> B(n);
    for (int& value : B) {
        value = dis(gen);
    }

    start = chrono::high_resolution_clock::now();
    sort(B.begin(), B.end());
    end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> stdSort_time = end - start;
    cout << "Стандартная сортировка заняла " << stdSort_time.count() << " мс.\n";

    return 0;
}
