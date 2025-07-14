#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include<sstream>

using namespace std;

double to_num(const string& s) {
    double num;
    stringstream ss(s);
    ss >> num;
    return num;
}


string* ReadData (ifstream &fin, string names[7]){
	string line;
    if (getline(fin, line)) {  // read one full line from the file
        stringstream ss(line);
        string value;
        int i = 0;
        while (getline(ss, value, ',') && i < 7) {
            names[i++] = value;
        }
    }
return names;
}


double* Survivability(ifstream &fin, double attributes[1000][7]) {
    string line;
    int rows = 0;
    while (getline(fin, line) && rows < 1000) {
        stringstream ss(line);
        string token;
        int col = 0;
        while (getline(ss, token, ',') && col < 7) {
            if (!token.empty()) {
                attributes[rows][col] = to_num(token);
                col++;
            }
        }
        rows++;
    }
}

int gradient_descent(double attributes[1000][7], int m, int b){
	int y[713]={};
	int j[5]={};
	int M[713][6]={};
	
	 for (int i=0;i<712;i++){
		y[i]=attributes[i][6];
		M[i][0]=1;
		
		for (int l=1;l<6;l++){
		M[i][l] = attributes[i][l];
	}
	//	cout << M[i][5] << endl; //for testing purposes
	
	
	
	const int points=713;
	const double alpha = 0.01;
	const int iterations = 1000;
	const int features = 5;
	
	double theta[features + 9]={};
	
	
	for (int i =0; i <713; i++){
		double prediction = 0.0;
		// Perform Gradient Descent
    for (int iter = 0; iter < iterations; iter++) {
        double gradients[features + 1] = {};

        // Compute gradients
        for (int i = 0; i < points; i++) {
            // Compute hypothesis for row i
            double prediction = 0.0;
            for (int j = 0; j <= features; j++) {
                prediction += theta[j] * M[i][j];
            }

            double error = prediction - y[i];

            // Accumulate gradient for each theta
            for (int j = 0; j <= features; j++) {
                gradients[j] += error * M[i][j];
            }
        }

        // Average and update theta
        for (int j = 0; j <= features; j++) {
            gradients[j] /= points;
            theta[j] -= alpha * gradients[j];
        }
    }
		
	}
	

	return theta[1];
	
	}
	
}


/*
int gradient_descent(double attributes[1000][7], int m, int b) {
    const int features = 5;        // Number of features (excluding intercept)
    const int points = 713;        // Total number of training points
    const double alpha = 0.01;     // Learning rate
    const int iterations = 1000;   // Number of iterations

    // Initialize y and feature matrix X (M)
    double y[points] = {};
    double M[points][features + 1] = {}; // +1 for intercept
    double theta[features + 1] = {};     // Parameters to learn

    // Fill in M and y
    for (int i = 0; i < points; i++) {
        y[i] = attributes[i][6];
        M[i][0] = 1.0; // Intercept term

        for (int j = 1; j <= features; j++) {
            M[i][j] = attributes[i][j]; // Copy features
        }
    }

    // Perform Gradient Descent
    for (int iter = 0; iter < iterations; iter++) {
        double gradients[features + 1] = {};

        // Compute gradients
        for (int i = 0; i < points; i++) {
            // Compute hypothesis for row i
            double prediction = 0.0;
            for (int j = 0; j <= features; j++) {
                prediction += theta[j] * M[i][j];
            }

            double error = prediction - y[i];

            // Accumulate gradient for each theta
            for (int j = 0; j <= features; j++) {
                gradients[j] += error * M[i][j];
            }
        }

        // Average and update theta
        for (int j = 0; j <= features; j++) {
            gradients[j] /= points;
            theta[j] -= alpha * gradients[j];
        }
    }

    // Output the final theta values
    cout << "Learned parameters after " << iterations << " iterations:\n";
    for (int j = 0; j <= features; j++) {
        cout << "theta[" << j << "] = " << fixed << setprecision(4) << theta[j] << endl;
    }

    return 0;
}
*/



int main(){
	
	int m=8,b=3;
	ifstream fin("training_data.csv");
	
		if (!fin.is_open()) {
        cerr << "Error opening file!" << endl;
        return 1;
    }
    
    
	string names[7];
	ReadData(fin, names);
	double attributes[1000][7]={0};
	Survivability(fin, attributes);
	
	for (int i=0;i<100;i++){
		for (int j=0;j<7;j++){
			cout << attributes[i][j] << " ";
		}
		cout << endl;
	}

	
cout << gradient_descent(attributes,m,b);

return 0;
}


