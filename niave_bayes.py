# -*- coding: utf-8 -*-
import sys
import numpy as np
import math

#args: [<training_file>, <test_file>]
args = sys.argv
training_file = args[1]
test_file = args[2]

training_data = []
num_observations = 0
num_classes = 0

def main():
    training_data = loadFile(training_file)
    training_data = sorted(training_data, key=lambda x: x[-1])
    training_data = np.vstack(training_data)
    
    global num_classes
    num_classes = training_data[-1][-1] + 1
    
    global num_observations
    num_observations = np.shape(training_data)[0]
    
    classes = separateClasses(training_data)
    
    test_data = loadFile(test_file)
    gaussians = getAllClassGaussians(classes)
    
    predictions = makePredictions(classes, gaussians, test_data)
    acc = checkAccuracy(test_data[:,-1], predictions)
    
    print("Classification Accuracy = ", acc)
    
def checkAccuracy(real, predictions):
    right = 0
    for i in range(len(real)):
        if (real[i] == predictions[i]):
            right += 1
    return right / len(real)
      
def makePredictions(classes, gaussians, test_data):
    observations = []
    for i in range(len(test_data)):
        obs = test_data[i]
        real = obs[-1]
        pred = calcClass(classes, gaussians, obs[:-1])
        observations.append(pred[1])
        acc = 0
        if (pred[1] == real):
            acc = 1
        
        print("ID = {:d}, Predicted = {:d}, Probability = {:e}, True = {:d}, Accuracy = {:d}".format(i+1, pred[1], pred[0][pred[1]], real, acc))
    
    return observations
        
def calcClass(classes, class_gaussians, observation):
    prob = {}
    
    for class_num, gaussians in class_gaussians.items():
        prob[class_num] = 1
        for i in range(len(gaussians)):
            mean, std = gaussians[i]
            x = observation[i]
            prob[class_num] *= calculate(x, mean, std)
    
    amax = argmax(prob)
    return (prob, amax)

def argmax(d):
    arg = -1
    highest = -1
    for k, v in d.items():
        if (v >= highest):
            highest = v
            arg = k
    return arg
      
def probabilityClass(classes, class_num):
    return np.shape(classes[class_num])[0] / num_observations

def getAllClassGaussians(classes):
    gaussians = {}
    global num_classes
    
    for cl in range(0, num_classes):
        gaussians[cl] = getClassGaussians(classes, cl)

    
    return gaussians

# each list is mean, stdev of that attribute / column
def getClassGaussians(classes, class_num):
    class_gaussians = []
    for dimension in range(np.shape(classes[class_num])[1] - 1):
        gauss = getGaussian(classes[class_num], dimension)
        class_gaussians.append(gauss)
        #print out here so i dont have to rewrite the code :)
        print("Class {:d}, attribute {:d}, mean = {:f}, std = {:f}".format(class_num, dimension, gauss[0], gauss[1]))
    return class_gaussians

def getGaussian(class_object, dimension):
    return (np.mean(class_object[:,dimension]), np.std(class_object[:,dimension]))
    
def separateClasses(data):
    separated = {}
    global num_classes
    
    last = 0
    for i in range(num_classes):
        size = np.where(data[:,-1] == i)[0][-1]
        separated[i] = data[last:size]
        last = size + 1
    return separated

def calculate(x, mean, std):
    std = max(std, 0.01)
    exp = math.exp(-math.pow(x - mean, 2)/(2*math.pow(std, 2)))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exp

def loadFile(file_name):
    with open(file_name) as f:
        data = []
        for line in f:
            data.append(list(map(int, line.split())))
            
        data = np.asarray(data)
        return data


main()