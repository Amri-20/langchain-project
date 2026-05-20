
#include <vector>
using namespace std;

class BeingZero {
public:
    vector<int> solve(int n, vector<vector<int>>& matrix, int q, vector<vector<int>>& queries) {
        vector<vector<vector<int>>> pref(11, vector<vector<int>>(n + 1, vector<int>(n + 1, 0)));
        
        // Build the 2D prefix sum arrays for each value from 1 to 10
        for (int v = 1; v <= 10; ++v) {
            for (int i = 1; i <= n; ++i) {
                for (int j = 1; j <= n; ++j) {
                    // Check if current matrix element (0-indexed) matches 'v'
                    int match = (matrix[i - 1][j - 1] == v) ? 1 : 0;
                    
                    // 2D Prefix sum formula
                    pref[v][i][j] = pref[v][i - 1][j] + pref[v][i][j - 1] - pref[v][i - 1][j - 1] + match;
                }
            }
        }
        
        vector<int> ans;
        
        // Process each query
        for (int i = 0; i < q; ++i) {
            int x1 = queries[i][0];
            int y1 = queries[i][1];
            int x2 = queries[i][2];
            int y2 = queries[i][3];
            
            int distinct_count = 0;
            for (int v = 1; v <= 10; ++v) {
                int count = pref[v][x2][y2] - pref[v][x1 - 1][y2] - pref[v][x2][y1 - 1] + pref[v][x1 - 1][y1 - 1];
                if (count > 0) {
                    distinct_count++;
                }
            }
            ans.push_back(distinct_count);
        }
        return ans;
    }
};