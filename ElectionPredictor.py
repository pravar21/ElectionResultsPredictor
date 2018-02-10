import pandas as pd 
import numpy as np
import random
import time
import math

#Data with features and target values
#Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
#Helper functions are provided so you shouldn't need to learn Pandas
dataset = pd.read_csv("data.csv")

#========================================== Data Helper Functions ==========================================

#Normalize values between 0 and 1
#dataset: Pandas dataframe
#categories: list of columns to normalize, e.g. ["column A", "column C"]
#Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData

#Encode categorical values as mutliple columns (One Hot Encoding)
#dataset: Pandas dataframe
#categories: list of columns to encode, e.g. ["column A", "column C"]
#Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
	return pd.get_dummies(dataset, columns=categories)

#Split data between training and testing data
#dataset: Pandas dataframe
#ratio: number [0, 1] that determines percentage of data used for training
#Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
	tr = int(len(dataset)*ratio)
	return dataset[:tr], dataset[tr:]

#Convenience function to extract Numpy data from dataset
#dataset: Pandas dataframe
#Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
	features = dataset.drop(["can_id", "can_nam","winner"], axis=1).values
	labels = dataset["winner"].astype(int).values
	return features, labels

#Convenience function to extract data from dataset (if you prefer not to use Numpy)
#dataset: Pandas dataframe
#Return: features list and corresponding labels as a list
def getPythonList(dataset):
	f, l = getNumpy(dataset)
	return f.tolist(), l.tolist()

#Calculates accuracy of your models output.
#solutions: model predictions as a list or numpy array
#real: model labels as a list or numpy array
#Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / float(labels.size)

#===========================================================================================================

class KNN:
    def __init__(self):
		#KNN state here
		#Feel free to add methods
        self.trainingDataset = []
        return

    def preprocessKNN(self,partialDataset):
        categories = ["net_ope_exp","net_con","tot_loa"]
        normDataset = normalizeData(partialDataset,categories)
        categories = ["can_off","can_inc_cha_ope_sea"]
        oneHotEncodedDataset = encodeData(normDataset,categories)
        return getNumpy(oneHotEncodedDataset)

    def getMajorityVote(self,neighbors):
        winnerVotes = {}

        for x in range(len(neighbors)):
            response = neighbors[x][1]

            if response in winnerVotes:
                winnerVotes[response] += 1
            else:
                winnerVotes[response] = 1

        sortedWinnerVotes = sorted(winnerVotes.iteritems(), key=lambda (k,v): (v,k) , reverse=True)

        return sortedWinnerVotes[0][0]

    def calculateEuclideanDistance(self,tuple1, tuple2):
        distance = 0

        for x in range(len(tuple1)):
            distance += pow((tuple1[x] - tuple2[x]), 2)
        return np.math.sqrt(distance)

    def getKNearestNeighbors(self, testFeatureTuple, k):
        distances = []
        neighbors = []

        for x in range(len(self.trainingFeatures)):
            distance = self.calculateEuclideanDistance(testFeatureTuple, self.trainingFeatures[x])
            distances.append(((self.trainingFeatures[x],self.trainingLabels[x]), distance))

        distances.sort(key = lambda x : x[1])

        for x in range(k):
            neighbors.append(distances[x][0])

        return neighbors

    def train(self, features, labels):
        #training logic here
		#input is list/array of features and labels
        self.trainingFeatures = features
        self.trainingLabels = labels
        return

    def predict(self, features):
        #Run model here
		#Return list/array of predictions where there is one prediction for each set of features
        k = 3
        predictions = []

        for x in range(len(features)):
            kNearestNeighbors = self.getKNearestNeighbors(features[x], k)
            majorityVote = self.getMajorityVote(kNearestNeighbors)
            predictions.append(majorityVote)

        return predictions

class Perceptron:
    def __init__(self):
        #Perceptron state here
		#Feel free to add methods
        self.weights = []
        self.bias = 0.0
        return

    def preprocessPerceptron(self,partialDataset):
        categories = ["net_ope_exp","net_con","tot_loa"]
        normDataset = normalizeData(partialDataset,categories)
        categories = ["can_off","can_inc_cha_ope_sea"]
        oneHotEncodedDataset = encodeData(normDataset,categories)
        return getNumpy(oneHotEncodedDataset)

    def stepActivation(self,summation):
        if summation >= 0:
            return 1
        else:
            return 0

    def getPredictionFromPerceptron(self,feature):
        summation = np.dot(feature, self.weights.reshape(9, 1)) + self.bias
        return self.stepActivation(summation)

    def train(self, features, labels):
        #training logic here
		#input is list/array of features and labels
        t1 = time.time()
        randomWeights = []
        self.bias = 0.01
        learningRate = 0.01

        for x in range(0,9):
            randomWeights.append(random.uniform(-0.1,0.1))

        self.weights = np.array(randomWeights)
        count = 0
        while time.time() - t1 < 60:
            for x in range(len(features)):
                if count == len(features):
                    break

                predictedOutput = self.getPredictionFromPerceptron(features[x])
                actualOutput = labels[x]

                if predictedOutput != actualOutput:
                    if actualOutput == 0:
                        actualOutput = -1
                    self.weights = self.weights + learningRate * actualOutput * features[x]
                    self.bias = self.bias + learningRate * actualOutput
                    count = 0
                else:
                    count = count + 1

        return

    def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
        predictions = []

        for x in range(len(features)):
            predictions.append(self.getPredictionFromPerceptron(features[x]))

        return predictions

class MLP:
    def __init__(self):
		#Multilayer perceptron state here
		#Feel free to add methods
        return

    def preprocessMLP(self,partialDataset):
        categories = ["net_ope_exp","net_con","tot_loa"]
        normDataset = normalizeData(partialDataset,categories)
        categories = ["can_off","can_inc_cha_ope_sea"]
        oneHotEncodedDataset = encodeData(normDataset,categories)
        return getNumpy(oneHotEncodedDataset)

    def sigmoidActivation(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def stepActivation(self, summation):
        if summation >= 0:
            return 1
        else:
            return 0

    def getOutputFromLayer(self, feature,weights,bias):
        summation = np.dot(feature, weights) + bias
        return self.sigmoidActivation(summation)

    #topology of NN : 9 nodes in input layer, 9 in hidden layer, 1 in output
    def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels

        self.weights1 = 2 * np.random.random((9,9)) - 1
        self.weights2 = 2 * np.random.random((9,1)) - 1
        self.bias1 = np.zeros(shape=(9,1))
        self.bias2 = 0
        learningRate = 0.01

        count = 0
        t1 = time.time()
        while time.time() - t1 < 60:
            for x in range(len(features)):
                if count == len(features):
                    break

                outputFromHiddenLayer = self.getOutputFromLayer(features[x],self.weights1,self.bias1.reshape(1,9))  #1*9
                predictedOutput = self.getOutputFromLayer(outputFromHiddenLayer,self.weights2,self.bias2) #1*1
                actualOutput = labels[x]

                if predictedOutput != actualOutput:

                    outputLayerError = actualOutput - predictedOutput
                    outputLayerDelta = outputLayerError * self.sigmoidDerivative(predictedOutput)#1*1

                    hiddenLayerError = np.dot(self.weights2,outputLayerDelta) #9*1 * 1*1 = 9*1 matrix
                    hiddenLayerDelta = hiddenLayerError * self.sigmoidDerivative(outputFromHiddenLayer.T) # 9*1

                    outputLayerAdjustment = outputFromHiddenLayer * outputLayerDelta #1*9
                    hiddenLayerAdjustment = hiddenLayerDelta.dot(features[x].reshape(1,9)) # 9*9

                    # Updating the weights.
                    self.weights2 += outputLayerAdjustment.reshape(9,1) * learningRate
                    self.weights1 += hiddenLayerAdjustment * learningRate

                    # Updating the bias
                    self.bias1 = self.bias1 + learningRate * hiddenLayerDelta
                    self.bias2 = self.bias2 + learningRate * outputLayerDelta


                    count = 0
                else:
                    count = count + 1
        return

    def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
        predictions = []

        for x in range(len(features)):
            outputFromHiddenLayer = self.getOutputFromLayer(features[x], self.weights1, self.bias1.reshape(1,9))  # 1*9
            predictedOutput = self.getOutputFromLayer(outputFromHiddenLayer, self.weights2, self.bias2)  # 1*1

            if(predictedOutput[0] > 0.5) :
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

class ID3:
    def __init__(self):
		#Decision tree state here
		#Feel free to add methods
        return

    def splitIntoEqualBuckets(self,values, buckets):
        a = (values.size / float(buckets)) * np.arange(1, buckets + 1)
        processed = a.searchsorted(np.arange(values.size))
        return processed[values.argsort().argsort()]

    def splitAttributesIntoBuckets(self,normDataset,categories):
        for category in categories:
            values = normDataset[category].values
            normDataset[category] = self.splitIntoEqualBuckets(values, 5)

        return normDataset

    def preprocessID3(self,partialDataset):
        categories = ["net_ope_exp","net_con","tot_loa"]
        normDataset = normalizeData(partialDataset,categories)
        bucketedDataSet = self.splitAttributesIntoBuckets(normDataset,categories)
        features, labels = getNumpy(bucketedDataSet)
        return features,labels

    def getMajorityValue(self,labels):
        valueFrequencies = {}
        for label in labels:
            if (valueFrequencies.has_key(label)):
                valueFrequencies[label] += 1.0
            else:
                valueFrequencies[label] = 1.0

        if len(valueFrequencies) == 1:
            return valueFrequencies.keys()[0]

        if valueFrequencies[0] > valueFrequencies[1]:
            return 0
        else:
            return 1

    def getEntropy(self, features, labels):
        if features.shape[0] ==0:
            return 0

        entropy = 0.0
        valueFrequencies = {}

        for label in labels:
            if (valueFrequencies.has_key(label)):
                valueFrequencies[label] += 1.0
            else:
                valueFrequencies[label] = 1.0

        for frequency in valueFrequencies.values():
            entropyVal = (-frequency / features.shape[0]) * math.log(frequency / features.shape[0], 2)
            entropy += entropyVal

        return entropy

    def getGain(self,features, attr, labels):

        entropyOfSubset = 0.0
        valueFrequencies = {}

        for feature in features:
            if (valueFrequencies.has_key(feature[attr])):
                valueFrequencies[feature[attr]] += 1.0
            else:
                valueFrequencies[feature[attr]] = 1.0

        for val in valueFrequencies.keys():
            probability = valueFrequencies[val] / sum(valueFrequencies.values())
            listOfTuples = [(features[x],labels[x]) for x in range(len(features)) if features[x][attr] == val]
            dataSubset = []
            labelSubset = []
            for tuple in listOfTuples:
                dataSubset.append(tuple[0])
                labelSubset.append(tuple[1])

            dataSubset = np.array(dataSubset)
            entropyOfSubset += probability * self.getEntropy(dataSubset, labelSubset)

        return (self.getEntropy(np.array(features), labels) - entropyOfSubset)

    def getBestAttribute(self,features, attributes, labels):
        gainValues = []

        for attribute in attributes:
            gainValues.append((self.getGain(features,attribute,labels),attribute))

        gainValues.sort(reverse=True)
        return gainValues[0][1]

    def getUniqueValues(self,features, best):
        uniqueValues = {}

        for feature in features:
            if not uniqueValues.has_key(feature[best]):
                uniqueValues[feature[best]]=1
            else:
                continue

        return uniqueValues.keys()

    def getTuples(self,features, best, val,labels):
        listOfTuples = [(features[x], labels[x]) for x in range(len(features)) if features[x][best] == val]
        featureSubset = []
        labelSubset = []
        for tuple in listOfTuples:
            featureSubset.append(tuple[0])
            labelSubset.append(tuple[1])
        return featureSubset,labelSubset

    def createDecisionTree(self,features, attributes, labels):
        defaultMajorityValue = self.getMajorityValue(labels)

        if len(features)== 0 or len(attributes) <= 0:
            return defaultMajorityValue

        unique, counts = np.unique(labels, return_counts=True)
        valueToCountMapping = dict(zip(unique, counts))

        if valueToCountMapping[labels[0]] == len(labels):
            return labels[0]

        else:
            bestAttribute = self.getBestAttribute(features, attributes, labels)

            if bestAttribute==0:
                tree = {"net_ope_exp": {}}
            elif bestAttribute==1:
                tree = {"net_con": {}}
            elif bestAttribute==2:
                tree = {"tot_loa": {}}
            elif bestAttribute == 3:
                tree = {"can_off": {}}
            else:
                tree = {"can_inc_cha_ope_sea": {}}

            for val in self.getUniqueValues(features, bestAttribute):
                featureSubset,labelSubset = self.getTuples(features, bestAttribute, val,labels)
                subtree = self.createDecisionTree(featureSubset,[attr for attr in attributes if attr != bestAttribute],labelSubset)

                if bestAttribute == 0:
                    tree["net_ope_exp"][val] = subtree
                elif bestAttribute == 1:
                    tree["net_con"][val] = subtree
                elif bestAttribute == 2:
                    tree["tot_loa"][val] = subtree
                elif bestAttribute == 3:
                    tree["can_off"][val] = subtree
                else:
                    tree["can_inc_cha_ope_sea"][val] = subtree

        return tree

    def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
        attributes = [0,1,2,3,4]
        self.decisionTree = self.createDecisionTree(features, attributes, labels)
        return

    def traverseTree(self,d,feature,attributeNameToNumberMap,cellValue,check):
        if not isinstance(d, dict):
            self.predictions.append(int(d))
            return

        if check:
            if d.has_key(cellValue):
                v = d[cellValue]
                self.traverseTree(v,feature,attributeNameToNumberMap,None,False)
            else:
                self.predictions.append(0)
        else:
            v = d[d.keys()[0]]
            cellValue = feature[attributeNameToNumberMap[d.keys()[0]]]
            if isinstance(v, dict):
                self.traverseTree(v,feature,attributeNameToNumberMap,cellValue,True)
        return

    def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
        self.predictions = []
        print pd.json.dumps(self.decisionTree, indent=4)
        attributeNameToNumberMap ={}
        attributeNameToNumberMap["net_ope_exp"] = 0
        attributeNameToNumberMap["net_con"] = 1
        attributeNameToNumberMap["tot_loa"] = 2
        attributeNameToNumberMap["can_off"] = 3
        attributeNameToNumberMap["can_inc_cha_ope_sea"] = 4

        for x in range(len(features)):
            self.traverseTree(self.decisionTree,features[x],attributeNameToNumberMap,None,False)

        return self.predictions


kNN = KNN()
train_ratio = 0.7
train_dataset, test_dataset = trainingTestData(dataset, train_ratio)
train_features, train_labels = kNN.preprocess(train_dataset)
kNN.train(train_features, train_labels)
test_features, test_labels = kNN.preprocess(test_dataset)
predictions = kNN.predict(test_features)
accuracy = evaluate(predictions, test_labels)