#include <iostream>

using namespace std;

int main()
{
    // We declare an array with a maximum size.
    // The user can enter dimensions up to 10x10.
    int matrix[10][10];
    int rows, cols;

    // --- Get Matrix Dimensions ---
    cout << "Enter the number of rows (up to 10): ";
    cin >> rows;
    cout << "Enter the number of columns (up to 10): ";
    cin >> cols;

    // --- Get Matrix Elements ---
    cout << "\nEnter the elements of the matrix:" << endl;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            cin >> matrix[i][j];
        }
    }

    // --- Display the Matrix ---
    cout << "\nThe matrix you entered is:" << endl;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // Print element followed by a tab for spacing
            cout << matrix[i][j] << "\t";
        }
        // Print a newline character after each row is complete
        cout << endl;
    }

    return 0;
}