#include <iostream>
#include <vector>

using namespace std;


int counting_sort (std::vector<int>& A, int K, int start, int end) {

    int ind = 0, sum = 0, i = start;
    vector <int> C(K, 0);

    for (int x : A ) C [ x ]++;

    for ( int x = 0; x < K ; ++ x )
        for ( int j = 0; j < C [ x ]; ++ j )
            A [ ind ++] = x ;
    
    for(; i <= end; i++) {
        sum += C[i];
    }

    return sum;
}

int main () {

    int n, i, q, l, r;
    cin >> n;

    vector<int> a(n, 0);

    for(; i < n; i++){
        a[i] = i;
        // cin >> a[i];
    } // считываем данные, но эту штуку закомментируйте, она только для меня будет. Для себя генерируйте рандомно

    cin >> q;

    while(q--){
        cin >> l >> r;
        cout << "Отрезку принадлежит: " << counting_sort(a, n, l, r) << '\n'; // ответ
    }

    return 0;
}