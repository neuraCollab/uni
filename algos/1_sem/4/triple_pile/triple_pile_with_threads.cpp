#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>

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
    for (int i = n / 3 - 1; i >= 0; i--)
        heapify(A, n, i);

    for (int i = n - 1; i > 0; i--) {
        swap(A[0], A[i]);
        heapify(A, i, 0);
    }
}

int main() {
    int n;
    cout << "Введите длину массива: ";
    cin >> n;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100);

    vector<int> A(n);
    for (int& value : A) {
        value = dis(gen);
    }

    auto start = chrono::high_resolution_clock::now();
    
    unsigned int num_threads = thread::hardware_concurrency();
    vector<thread> threads(num_threads);

    // Разбиваем задачу на несколько потоков
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads[i] = thread(ternaryHeapSort, ref(A));
    }

    // Дожидаемся завершения всех потоков
    for (thread &t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ternaryHeapSort_time = end - start;
    cout << "Тернарная пирамидальная сортировка заняла " << ternaryHeapSort_time.count() << " мс.\n";
    cout << "Я не понимаю почему с потоками выходит дольше чем без них? \n";
        

    return 0;
}
