import sys
import numpy as np
import math

from sklearn.naive_bayes import GaussianNB


def main():
    train_file = loadFile("Desktop/Fall2018/unsorted/train2.txt")
    test_file = loadFile("Desktop/Fall2018/unsorted/test2.txt")

    train_data = train_file[:, :-1]
    labels = train_file[:, -1]
    test_data = test_file[:, :-1]

    gnb = GaussianNB()
    gnb.fit(train_data, labels)

    pred = gnb.predict(test_data)

    print("Number of mislabeled points out of a total %d points : %d" %
          (train_data.shape[0], sum(i != j for i, j in zip(labels, pred))))


def loadFile(file_name):
    with open(file_name) as f:
        data = []
        for line in f:
            data.append(list(map(float, line.split())))

        data = np.asarray(data)
        return data


main()
