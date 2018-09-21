# -*- coding: utf-8 -*-
import sys
import numpy as np
import math

training_data = []
num_observations = 0
num_classes = 0


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

    correct = 0
    for i in range(np.shape(test_data)[0]):
        acc = 0
        obs = test_data[i]
        calc = calculateProbability(classes, gaussians, obs[:-1])
        if (calc[0] == int(obs[-1])):
            correct += 1
            acc = 1

        print("ID = {:5.0f}, Predicted = {:3.0f}, Probability = {:0.4f}, True = {:3.0f}, Accuracy = {:4.2f}".format(
            i+1, calc[0], calc[1], obs[-1], acc))

    print("Classification Accuracy = {:6.4f}".format(
        correct / np.shape(test_data)[0]))


def calculateProbability(classes, gaussians, obs):
    denom = 0
    for cl_num in classes.keys():
        denom += probGivenClass(obs, cl_num, gaussians) * \
            classFrequency(classes, cl_num)

    max_prob = -1
    max_class = 0
    for cl_num in classes.keys():
        prob = probGivenClass(obs, cl_num, gaussians) * \
            classFrequency(classes, cl_num)
        prob = prob / denom
        if (prob > max_prob):
            max_prob = prob
            max_class = cl_num
    return (max_class, max_prob)


def classFrequency(classes, cl_num):
    return np.shape(classes[cl_num])[0] / num_observations


def probGivenClass(obs, cl_num, gaussians):
    prob_density = 1
    for col in range(len(gaussians[cl_num])):
        dens = calculate(obs[col], gaussians[cl_num][col]
                         [0], gaussians[cl_num][col][1])
        prob_density *= dens
    return prob_density

# todo: divide each probability by probability of an observation in each class (SUM RULE)


def probabilityClass(classes, class_num):
    return np.shape(classes[class_num])[0] / num_observations


def getAllClassGaussians(classes):
    gaussians = {}

    for cl in np.unique(list(classes.keys())):
        gaussians[cl] = getClassGaussians(classes, cl)

    return gaussians

# each list is mean, stdev of that attribute / column


def getClassGaussians(classes, class_num):
    class_gaussians = []

    for dimension in range(np.shape(classes[int(class_num)])[1] - 1):
        gauss = getGaussian(classes[class_num], dimension)
        class_gaussians.append(gauss)
        # print out here so i dont have to rewrite the code :)
        print("Class {:.0f}, attribute {:.0f}, mean = {:0.2f}, std = {:0.2f}".format(
            class_num, dimension, gauss[0], gauss[1]))
    return class_gaussians


def getGaussian(class_object, dimension):
    return (np.mean(class_object[:, dimension]), np.std(class_object[:, dimension]))


def separateClasses(data):
    separated = {}
    global num_classes

    last = 0
    for i in np.unique(data[:, -1]):
        i = int(i)
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
