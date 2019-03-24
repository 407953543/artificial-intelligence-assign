import numpy as np
import pandas as pd
import random
import itertools

# make your random same as mine

def load_dataset():
    '''
    Load the train examples and labels from .csv file

    Arguments:
    rtype x: ndarray, (9, 49)
    rtype y: ndarray, (1, 49)
    '''
    train = pd.read_csv(filepath_or_buffer='training-set.csv', header=None).values
    x, y = train[:, :9].T, train[:, -1:].T
    return x, y


def initialize_generation():
    '''
    Initializa W and Theta
    W[i]: uniformly distributed in (0, 1)
    Theta[i]: "standard norm" + 2.25

    Arguments:
    rtype W: List[ndarray], len(W) == 100, W[i].shape = (1, 9)
    rtype Theta: List[float], len(Theta) == 100
    '''
    W = [np.random.uniform(-1, 1, (1, 9)) for _ in range(100)]
    Theta = [float(np.random.randn(1) + 2.25) for _ in range(100)]
    return W, Theta


def predict(x, y, w, theta):
    '''
    Calculate the accuracy of a certain w and theta

    Arguments:
    type x: ndarray, (9, 49)
    type y: ndarray, (1, 49)
    type w: ndarray, (1, 9)
    type theta: float

    rtype acc: float
    '''
    predict = np.matmul(w, x)

    # greater than theta, predict = 1, less than theta, predict = 0
    predict = np.where(predict > theta, 1, 0)

    acc = np.sum(predict == y) / y.shape[1]
    return acc


def random_choose(x, y, W, Theta, k, l):
    '''
    randomly select k elements out of the l
    then pick the element with highest predict accuracy among the selected k elements

    Arguments:
    type x: ndarray, shape(9, 49)
    type y: ndarray, shape(1, 49)
    type W: List[ndarray]
    type Theta: List[float]
    type k: int
    type l: List

    rtype: int
    '''
    random.shuffle(l)
    picked_index = l[:k]
    highest_acc, target_index = -1, -1
    for index in picked_index:
        acc = predict(x, y, W[index], Theta[index])
        if acc > highest_acc:
            highest_acc, target_index = acc, index
    return target_index


def mutation(w, theta):
    new_w = np.array(w)
    index = random.randint(0, w.shape[1] - 1)

    for i in range(w.shape[1]):
        if random.randint(1, 100) < 15:
            new_w[0, i] = float(np.random.uniform(-1, 1, 1))
    if random.randint(1, 100) < 50:
        theta = theta + float(np.random.randn(1))
    return new_w, theta

def update_parameters(x, y, W, Theta):
    tmp = list(range(100))
    new_W, new_Theta = [], []

    for _ in range(100):
        # copy operation, first 10 elements
        if _ < 10:
            index = random_choose(x, y, W, Theta, 7, tmp)
            w, theta = mutation(W[index], Theta[index])
            new_W.append(w); new_Theta.append(theta)

        # cross over operation, next 90 elements
        else:
            # choose the first parent for cross over
            index_1 = random_choose(x, y, W, Theta, 7, tmp)
            parent1_W = np.array(W[index_1])

            # choose the second parent for cross over
            tmp.remove(index_1)
            index_2 = random_choose(x, y, W, Theta, 7, tmp)
            parent2_W = np.array(W[index_2])
            tmp.append(index_1)

            # cross over and get the next generation
            i = random.randint(1, 10)
            parent1_W, parent2_W = np.concatenate([parent1_W[:, :i], parent2_W[:, i:]], axis=1), np.concatenate([parent2_W[:, :i], parent1_W[:, i:]],axis=1)
            
            if predict(x, y, parent1_W, Theta[index_2]) > predict(x, y, parent2_W, Theta[index_1]):
                w, theta = mutation(parent1_W, Theta[index_2])
                new_W.append(w); new_Theta.append(theta)
            else:
                w, theta = mutation(parent2_W, Theta[index_1])
                new_W.append(w); new_Theta.append(theta)

    return new_W, new_Theta

x, y = load_dataset()
W, Theta = initialize_generation()

for _ in range(500):
    highest_acc, target_index = -1, -1
    for i in range(100):
        acc = predict(x, y, W[i], Theta[i])
        if highest_acc < acc:
            highest_acc, target_index = acc, i
    W, Theta = update_parameters(x, y, W, Theta)

    print("iteration: {}, accuracy: {}%".format(_, highest_acc * 100))
    if highest_acc > 0.98:
        print(W[target_index], Theta[target_index])
        input()
        break
# update_parameters(x, y, W, Theta)

# print(len(W), W[1].shape, len(Theta), Theta[1])