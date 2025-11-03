#include <iostream>
#include <vector>
#include <cstdlib> // Для std::rand и std::srand
#include <ctime>   // Для std::time

int mergeAndCount(std::vector<int>& arr, int left, int mid, int right) {
    int count = 0;
    int i = left;
    int j = mid;
    int k = 0;
    std::vector<int> temp(right - left + 1);

    while (i < mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            count += mid - i; // Подсчет инверсий
        }
    }

    while (i < mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (i = left, k = 0; i <= right; i++, k++) {
        arr[i] = temp[k];
    }

    return count;
}

int mergeSortAndCount(std::vector<int>& arr, int left, int right) {
    int count = 0;
    if (left < right) {
        int mid = left + (right - left) / 2;

        count += mergeSortAndCount(arr, left, mid);
        count += mergeSortAndCount(arr, mid + 1, right);
        count += mergeAndCount(arr, left, mid + 1, right);
    }
    return count;
}

int main() {
    int n;
    std::cout << "Введите длину массива: ";
    std::cin >> n;

    std::vector<int> arr(n);
    std::srand(std::time(nullptr)); // Инициализация генератора случайных чисел

    // Заполнение массива случайными числами
    for (int& value : arr) {
        value = std::rand() % 100; // Случайное число от 0 до 99
    }

    std::cout << "Сгенерированный массив: ";
    for (int value : arr) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    int total_inversions = mergeSortAndCount(arr, 0, n - 1);
    std::cout << "Общее количество инверсий: " << total_inversions << std::endl;

    return 0;
}
