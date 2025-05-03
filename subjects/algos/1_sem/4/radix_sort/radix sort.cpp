#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

const int MAX_CHAR = 26; // количество букв в английском алфавите
// я не стал брать 256 как в кодировке, т к очень долго выполнялось
//  с большими числами, но если нужно поменяйте пожалуйста

void countingSort(vector<string>& arr, int index) {
    vector<string> output(arr.size());
    vector<int> count(MAX_CHAR + 1, 0);

    // Подсчет количества вхождений каждого символа
    for (const string& str : arr) {
        char ch = index < str.size() ? str[index] - 'a' : -1;
        count[ch + 1]++;
    }

    // Накопительный подсчет
    for (int i = 1; i < count.size(); i++) {
        count[i] += count[i - 1];
    }

    // Формирование отсортированного массива
    for (int i = arr.size() - 1; i >= 0; i--) {
        char ch = index < arr[i].size() ? arr[i][index] - 'a' : -1;
        output[count[ch + 1] - 1] = arr[i];
        count[ch + 1]--;
    }

    // Копирование обратно в исходный массив
    arr = output;
}

void radixSort(vector<string>& arr) {
    int maxLen = 0;
    for (const string& str : arr) {
        maxLen = max(maxLen, static_cast<int>(str.size()));
    }

    for (int i = maxLen - 1; i >= 0; i--) {
        countingSort(arr, i);
    }
}

int main() {
    int n;
    cout << "Введите количество строк: ";
    cin >> n;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100); // Длины строк от 1 до 100
    uniform_int_distribution<> char_dis(0, MAX_CHAR - 1);

    vector<string> arr(n);
    int totalLength = 0;

    for (string& str : arr) {
        int len = dis(gen);
        while (totalLength + len > 1e7) { // Проверка на условие суммарной длины
            len = dis(gen);
        }
        totalLength += len;
        str.resize(len);
        for (char& ch : str) {
            ch = 'a' + char_dis(gen);
        }
    }

    auto start = chrono::high_resolution_clock::now();
    radixSort(arr);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> radixSort_time = end - start;
    cout << "Поразрядная сортировка заняла " << radixSort_time.count() << " мс.\n";

    // Сравнение с std::sort
    vector<string> arr_copy = arr; // Копия для std::sort

    start = chrono::high_resolution_clock::now();
    sort(arr_copy.begin(), arr_copy.end());
    end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> stdSort_time = end - start;
    cout << "Стандартная сортировка заняла " << stdSort_time.count() << " мс.\n";

    return 0;
}
