#include <iostream>
#include <vector>
#include <cstdlib> // Для std::rand и std::srand
#include <ctime>   // Для std::time

void merge(std::vector<int>& A, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (A[i] <= A[j]) {
            temp[k++] = A[i++];
        } else {
            temp[k++] = A[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = A[i++];
    }

    while (j <= right) {
        temp[k++] = A[j++];
    }

    for (i = left, k = 0; i <= right; ++i, ++k) {
        A[i] = temp[k];
    }
}

void mergeSort(std::vector<int>& A, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(A, left, mid);
        mergeSort(A, mid + 1, right);
        merge(A, left, mid, right);
    }
}

void findPairsWithSum(const std::vector<int>& A, unsigned int S) {
    int left = 0;
    int right = A.size() - 1;
    bool found = false;

    while (left < right) {
        unsigned int currentSum = A[left] + A[right];
        
        if (currentSum < S) {
            left++;
        } else if (currentSum > S) {
            right--;
        } else {
            std::cout << "Пара найдена: (" << A[left] << ", " << A[right] << ")\n";
            left++;
            right--;
            found = true;
        }
    }

    if (!found) {
        std::cout << "Пары с суммой " << S << " не найдены.\n";
    }
}

int main() {
    int n;
    unsigned int S;
    
    std::cout << "Введите длину массива: ";
    std::cin >> n;
    
    std::cout << "Введите число S: ";
    std::cin >> S;

    std::vector<int> A(n);
    std::srand(static_cast<unsigned int>(std::time(nullptr))); // Инициализация генератора случайных чисел

    // Заполнение массива случайными числами
    for (int& value : A) {
        value = std::rand() % 100; // Случайное число от 0 до 99
    }

    // Сортировка массива слиянием
    mergeSort(A, 0, n - 1);

    // Вывод отсортированного массива
    std::cout << "Отсортированный массив: ";
    for (int value : A) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Поиск пар с суммой S
    findPairsWithSum(A, S);

    return 0;
}
