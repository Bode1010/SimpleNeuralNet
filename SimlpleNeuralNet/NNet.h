#ifndef NNet_H
#define NNet_H
#pragma once

#include "Matrix.h"//for matrix math
#include <math.h>//for activation functions
#include <thread>//For multithreading
#include <future>//for myltithreading
#include <string>
#include <utility>
#include <fstream>//for saving and loading
using namespace std;
typedef vector<Matrix> wArray;
typedef pair<float, float> doublef;

/**************************TODO************************************/
//graph of error over time normalized to fit the screen so we can see how good are network is doing. Maybe a viz subclass using SFML?
//Implement batchNorm
//make a way to switch btw dense and convolutional, apply accordingly during backprop

enum Activation {
	NONE, RELU, SIGMOID, TANH, SOFTMAX
};

enum LayerType {
	DENSE, GRU
};

struct OneOutput {
	OneOutput();
	OneOutput(float val1, int index1);
	float val;//value of the output
	int index;//which output it is
};

class Layer
{
public:
	Layer(LayerType Type, int Size, Activation, bool batchNorm);
	void print();
	/*The first float is it's preactivated value, the second is it's activated*/
	LayerType getType();
	float Activate(float x);
	float dActivate(float x);
	Activation getActivation();
	bool getBatchNorm();
	unsigned int size() { return mySize; }
	void operator=(const Layer &obj);

	Matrix neurons;

	//BatchNorm Variables
	doublef bn; //bn.First is gamma in f(x) = normalizedX * gamma + beta. bn.second is beta 
	vector<float> mean;
	vector<float> stdDev;

	//SoftMax vars
	float softMaxDenom = 0;

	//GRU vars: Neural networks and last hidden state
	Matrix prevNeurons, reset, update, hidden;
	vector<Matrix> resetPrevNW;//Weights for the previous state that enter the reset gate
	vector<Matrix> resetInputW;
	vector<Matrix> updatePrevNW;
	vector<Matrix> updateInputW;
	vector<Matrix> hiddenPrevNW;//Weights for the previous state that enter this hidden layer;
	vector<Matrix> hiddenInputW;
	float gr = 0.1; //growthrate

	/*Activations*/
	static float SigmoidActivate(float);
	static float SigmoidDActivate(float);
	static float ReLUActivate(float);
	static float ReLUDActivate(float);
	static float TanhActivate(float);
	static float TanhDActivate(float);
	float SoftMaxActivate(float);
	float SoftMaxDActivate(float);

private:
	unsigned int mySize;
	bool batchNorm;
	LayerType type;
	Activation activate;

};

class NNet
{
public:
	NNet() {};
	NNet(vector<Layer> layout);
	//trains with one output neuron instead of all of them.
	void trainWithOneOutput(vector<vector<float>> inputs, vector<OneOutput> outputs);
	//Updates the weights in the network
	void train(vector<vector<float>> inputs, vector<vector<float>> outputs);
	//Used for normal forward passes
	void feedForward(vector<float> input);
	//used to assign nets to other nets
	void operator=(const NNet &obj);
	//returns a vector of the output. Used after a forward pass (ie feedforward function)
	vector<float> getOutput();
	//returns the max output index after a forward pass
	int getMaxOutputIndex();
	//returns the max output after a forward pass
	float getMaxOutput();
	//Prints output in a straight line
	void printOutput();
	//Pretty self Explanatory tbh
	bool saveFilePresent();
	void save();
	void load();
	//Prints every weight. Used for debugging
	void visualize();
	~NNet();

private:
	//Default save file
	const string saveFile = "NNetSave.txt";

	//Holds the non activated values of the network
	vector<Layer> network;

	//returns the updates to be made after backpropping some inputs and outputs. returns changes to the weight array and the batch norm if applicable
	pair<wArray, vector<doublef>> FFandBPWithOneOutput(vector<float> input, OneOutput output);
	pair<wArray, vector<doublef>> FFandBP(vector<float> input, vector<float> output);

	//Different ways to forward pass
	Layer DenseFeedForward(Layer prevLayer, Layer curLayer, int curlayerID);
	Layer GRUFeedForward(Layer prevLayer, Layer curLayer, int curLayerID);
	//Different ways to backward pass
	void DenseBackPass(Layer& curLayer,Layer& nextLayer, int curLayerID, vector<Matrix>& change, vector<vector<float>>& temp, vector<doublef>& tempbn);
	void GRUBackPass(Layer& curLayer,Layer& nextLayer, int curLayerID, vector<Matrix>& change, vector<vector<float>>& temp, vector<doublef>& tempbn);

	//Used when training the network with threads
	void feedForwardTrain(vector<float> input, vector<Layer> &layers);

	//NOTE: weights for a certain layer are the ones in front of it
	//Holds weights and biases. Type of layer affects composition of weight matrix
	wArray weights;

	//How fast weights and biases update
	const float growthRate = 0.1;
};

#endif
