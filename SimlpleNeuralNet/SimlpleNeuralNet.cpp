#include "NNet.h"
#include <chrono>
using namespace std::chrono;

//NOTE: 
//I use stochastic gradient descent
//No batchNorm for last layer or first layer
//THE SPEED AT WHICH THE NETWORK TRAINS IS BASED ON THE NUMBER OF NEURONS, TOO LITTLE IT MIGHT NOT WORK/TAKE LONG
//TOO MUCH AND IT WILL WORK BUT TAKE FOREVER TO TRAIN
//JUST RIGHT AND IT WILL WORK AND TAKE THE LEAST TIME TO TRAIN, ESPECIALLY WITH BATCH NORM

//THINGS THAT CAUSE AVERAGING: NOTHING, YOU JUST NEED MORE TRAINING REPS!!!OMG!!!! IT IS IMPOSSIBLE TO BE TOO COMPLEX!!!!!!!!!
//Too few neurons
//too high growthrate:causes instability, not averaging
//Too many batchnormed layers: takes forever to train, not averaging

//Things that cause nan(ind)[ie, divide by zero error]:
//Too high growthrate when using batchnorm

//TODO:
//Implement lstm layers
//Implement batchnorm in train with one output function
//rmsprop, dropout
//Implement resnet and convolution
//Use encapsuation throughout my nnet.h
//Write a function to create a matrix by copying a row of another matrix, will reduce rutime. Will it tho?

//HOW TO USE BATCHNORM
//IF IT RETURNS NAN, REDUCE THE GROWTH RATE

//THROUGH TRIAL AND ERROR I HAVE DISCOVERED THE FOLLOWING:
//A NEURAL NETWORK IS A WAY TO CLASSIFY DATA IN A NON LINEAR WAY. IT IS LIKE DRAWING A LINE BETWEEN 2 GROUPS OF THINGS FOR EXAMPLE\
EXCEPT INSTEAD OF A STRAIGHT LINE IN A LINEAR FASHION, IT CAN BE A REALLY WIGGLY LINE TOO!
//I DONT KNOW WHAT ADDING ANOTHER LAYER DOES PER SE, BUT I KNOW THAT EACH NEURON BASICALLY REMOVES A "POWER" FROM ITS ACTIVATION AND DUPLICATES THAT ACTIVATION:\
FOR EXAMPLE, I ASSUME (ALMOST DEFINITELY INCORRECTLY) THAT TANH HAS THE SAME NUMBER OF "TURNS" THAT X^3 DOES(WHICH IS 2 TURNS),\
 BUT WHEN USED AS AN ACTIVATION FUNCTION, IT HAS THE SAME NUMBER OF TURNS THAT X^2 DOES (WHICH IS ONLY 1 TURN OR A PARABOLA),\
IF YOU DRAW THE GRAPH OF XOR YOU NEED AT LEAST 1 PARABOLLA OR 2 STRAIGHT LINES TO SEPERATE THE DATA INTO 2 PARTS, THEREFORE YOU CAN USE\
1 TANH ACTIVATED NEURONS OR 2 RELU ACTIVATED UNITS(IT HAS ONE "TURN" AND THAT BECOMES ZERO WHEN USED AS AN ACTIVATION FUNCTION, MAKING IT\
A STRAIGHT LINE) TO MAP AN XOR GATE. I THINK NONE ACTIVATION IS JUST A POINT IN SPACE, AND I AM PROBABLY WRONG ABOUT ALL THIS....\
BUT IF I TRY TO USE ANY LESS NEURONS, MY NEURAL NET DOEST WORK :/\
BATCH NORM ADDS A "TURN" TO WHICHEVER FUNCTION ITS APPLIED TO, IT GIVES AN ACTIVATED TANH NEURON 2 TURNS INSTEAD OF ONE, WHICH CAN ALSO \
CLASSIFY AN XOR GATE, ALTHOUGH UNECESSARY.

//FOR SOME REASON, ADDING MORE NEURONS WHEN USIN TANH DOESNT CHANGE THE NETWORK MUCH, BUT DOING THE SAME WITH THE RELU LAYER CAUSES AVERAGING\

int main()
{
	vector<Layer> layout;
	layout.push_back(Layer(DENSE, 2, NONE, false));//input layer
	layout.push_back(Layer(GRU, 2, NONE, false));//hidden layer
	layout.push_back(Layer(DENSE, 1, NONE, false));//output layer

	vector<float> input0;
	input0.push_back(0);
	input0.push_back(0);
	vector<float> input1;
	input1.push_back(1);
	input1.push_back(0);
	vector<float> input2;
	input2.push_back(0);
	input2.push_back(1);
	vector<float> input3;
	input3.push_back(1);
	input3.push_back(1);

	vector<vector<float>> inputBatch;
	inputBatch.push_back(input0);
	inputBatch.push_back(input1);
	inputBatch.push_back(input2);
	inputBatch.push_back(input3);

	vector<float> output0;
	output0.push_back(0);
	vector<float> output1;
	output1.push_back(1);
	vector<float> output2;
	output2.push_back(1);
	vector<float> output3;
	output3.push_back(0);

	vector<vector<float>> outputBatch;
	outputBatch.push_back(output0);
	outputBatch.push_back(output1);
	outputBatch.push_back(output2);
	outputBatch.push_back(output3);

	NNet myNet(layout);

	auto startTime = high_resolution_clock::now();

	char ans = 'O';
	int numReps = 4000;
	if (myNet.saveFilePresent()) {
		while (ans != 'Y' && ans != 'N') {
			system("cls");
			cout << "Load trained network? Y/N" << endl;

			cin >> ans;
			if (ans == 'Y') myNet.load();
			else if (ans == 'N') {
				startTime = high_resolution_clock::now();
				//Train with how many batches/
				for (int i = 1; i <= numReps; i++) {
					//Program timer/
					myNet.train(inputBatch, outputBatch);
					if (numReps > 10) {
						if (i % (numReps / 10) == 0) {
							cout << i / (numReps / 10);
							auto curTime = high_resolution_clock::now();
							auto duration = duration_cast<microseconds>(curTime - startTime);
							cout << ": " << duration.count() / 1000000.0f << " s" << endl;
							//myNet.feedForward(input0);
							//myNet.printOutput();
						}
					}
				}
				auto endTime = high_resolution_clock::now();
				auto duration = duration_cast<microseconds>(endTime - startTime);
				cout << "Training time with threads: " << duration.count() / 1000000.0f << endl;
				myNet.save();
			}
		}
	}
	else {
		//Train with how many batches/
		int numReps = 1;
		for (int i = 0; i < numReps; i++) {
			//Program timer/
			if (numReps > 10) {
				if (i % (numReps / 10) == 0) {
					cout << i / (numReps / 10);
					auto curTime = high_resolution_clock::now();
					auto duration = duration_cast<microseconds>(curTime - startTime);
					cout << ": " << duration.count() / 1000000.0f << " s" << endl;
				}
			}
			myNet.train(inputBatch, outputBatch);
		}
		auto endTime = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(endTime - startTime);
		cout << "Training time with threads: " << duration.count() / 1000000.0f << endl;
		cout << endl;
		myNet.save();
	}

	cout << "Insert 2 numbers to test the net, inserting -1 closes the program:" << endl;
	cout << endl;

	/*Test Net with input*/
	float a = 0;
	float b = 0;
	while (a != -1) {
		cin >> a;
		if (a == -1) break;
		cin >> b;
		cout << a << " " << b << ": ";
		vector<float> result = { a, b };
		myNet.feedForward(result);
		myNet.printOutput();
	}


	cout << "Program Finished. Hit return to close." << endl;
	cin.get();
	return 0;
}