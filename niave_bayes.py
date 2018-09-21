# -*- coding: utf-8 -*-
import sys
import numpy as np
import math

# args: [<training_file>, <test_file>]
args = sys.argv
training_file = args[1]
testing_file = args[2]

training_data = []
num_observations = 0
num_classes = 0


def main():
    naive_bayes(training_file, testing_file)


def naive_bayes(train_file, test_file):
    training_data = loadFile(train_file)
    training_data = sorted(training_data, key=lambda x: x[-1])
    training_data = np.vstack(training_data)

    global num_classes
    num_classes = int(training_data[-1][-1] + 1 - training_data[0][-1])

    global num_observations
    num_observations = np.shape(training_data)[0]

    classes = separateClasses(training_data)

    test_data = loadFile(test_file)
    gaussians = getAllClassGaussians(classes)

    

    calculateProbability(classes, gaussians, test_data[0])

    #predictions = makePredictions(classes, gaussians, test_data)
    #acc = checkAccuracy(test_data[:, -1], predictions)

    #print("Classification Accuracy = ", acc)

def calculateProbability(classes, gaussians, obs):
    denom = 0
    for cl_num in classes.keys():
        denom += probGivenClass(obs, cl_num, gaussians) * classFrequency(classes, cl_num)

    max_prob = -1
    max_class = 0
    for cl_num in classes.keys():
        prob = probGivenClass(obs, cl_num, gaussians) * classFrequency(classes, cl_num)
        prob = prob / denom
        if (prob > max_prob):
            max_prob = prob
            max_class = cl_num

def classFrequency(classes, cl_num):
    return np.shape(classes[cl_num])[0] / num_observations

def probGivenClass(obs, cl_num, gaussians):
    prob_density = 1
    for col in range(len(gaussians[cl_num])):
        dens = calculate(obs[col], gaussians[cl_num][col][0], gaussians[cl_num][col][1])
        prob_density *= dens
    return prob_density

def checkAccuracy(real, predictions):
    right = 0
    for i in range(len(real)):
        if (real[i] == predictions[i]):
            right += 1
    return right / len(real)

#todo: divide each probability by probability of an observation in each class (SUM RULE)
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

        print("ID = {:.0f}, Predicted = {:.0f}, Probability = {:e}, True = {:.0f}, Accuracy = {:.0f}".format(
            i+1, pred[1], pred[0][pred[1]], real, acc))

    return observations


def calcClass(classes, class_gaussians, observation):
    prob = {}

    den = 1
    for class_num in list(class_gaussians.keys()):
        den *= probabilityClass(classes, class_num)

    for class_num, gaussians in class_gaussians.items():
        prob[class_num] = 1
        for i in range(len(gaussians)):
            mean, std = gaussians[i]
            x = observation[i]
            calc = calculate(x, mean, std)
            # if (calc > 1):
            #print(class_num, i, x, gaussians[i], calc)
            prob[class_num] *= calc
        prob[class_num] *= probabilityClass(classes, class_num) / den

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

    start = list(classes.keys())[0]
    end = list(classes.keys())[-1]
    for cl in range(start, end + 1):
        gaussians[cl] = getClassGaussians(classes, cl)

    return gaussians

# each list is mean, stdev of that attribute / column


def getClassGaussians(classes, class_num):
    class_gaussians = []
    for dimension in range(np.shape(classes[int(class_num)])[1] - 1):
        gauss = getGaussian(classes[class_num], dimension)
        class_gaussians.append(gauss)
        # print out here so i dont have to rewrite the code :)
        print("Class {:.0f}, attribute {:.0f}, mean = {:f}, std = {:f}".format(
            class_num, dimension, gauss[0], gauss[1]))
    return class_gaussians


def getGaussian(class_object, dimension):
    return (np.mean(class_object[:, dimension]), np.std(class_object[:, dimension]))


def separateClasses(data):
    separated = {}
    global num_classes

    last = 0
    start = data[0][-1]
    end = data[-1][-1]
    for i in range(int(start), int(end + 1)):
        print(i)
        size = np.where(data[:, -1] == i)[0][-1]
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
            data.append(list(map(float, line.split())))

        data = np.asarray(data)
        return data


main()
