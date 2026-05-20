#include <iostream>
using namespace std;

int main()
{
    int m, n;
    int arr[100][100];
    cout << "Enter the value of row and column" << endl;
    cin >> m >> n;
    cout << "Enter the data into matrix" << endl;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cin >> arr[i][j];
        }
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << arr[i][j] << "\t";
        }
    }

    return 0;
}