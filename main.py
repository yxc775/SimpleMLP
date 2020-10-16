import random
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_sample(a=0.5, b=0.6, r=0.4):
    sample_list = []
    for _ in range(1000000):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        z = 0
        res = (x - a) ** 2 + (y - b) ** 2
        if res < r ** 2:
            z = 1
        item = [x, y, z]
        sample_list.append(item)
    return sample_list

def get_sample_larger(a=0.5, b=0.6, r=0.4):
    sample_list = []
    for _ in range(100):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        z = 0
        if x > y:
            z = 1
        item = [x, y, z]
        sample_list.append(item)
    return sample_list

def get_predict(x, y, w1, w2, w3, w4, w5, w6, b1, b2, b3):
    n1_out = sigmoid(w1 * x + w2 * y + b1)
    n2_out = sigmoid(w3 * x + w4 * y + b2)
    return stepF(w5 * n1_out + w6 * n2_out + b3)

def stepF(x):
    if x > 0:
        return 1
    else:
        return 0

def dstepF(x):
    if x > 0:
        return 1
    else:
        return 0

def update_weights(predict, answer, x, y, w1, w2, w3, w4, w5, w6, b1, b2, l):
    dE = -(answer - predict)
    loss = dE ** 2 * 0.5
    n1outtemp = sigmoid(w1 * x + w2 * y + b1)
    n2outtemp = sigmoid(w3 * x + w4 * y + b2)
    doutn1 = n1outtemp * (1 - n1outtemp)
    doutn2 = n2outtemp * (1 - n2outtemp)
    neww1 = w1 - l * dE * predict * w5 * doutn1 * x
    neww2 = w2 - l * dE * predict * w5 * doutn2 * y
    neww3 = w3 - l * dE * predict * w6 * doutn1 * x
    neww4 = w4 - l * dE * predict * w6 * doutn2 * y
    neww5 = w5 - l * dE * predict * n1outtemp
    neww6 = w6 - l * dE * predict * n2outtemp
    return neww1, neww2, neww3, neww4, neww5, neww6, loss


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    w1 = 0.5
    w2 = 0.1
    w3 = 11
    w4 = 0.2
    w5 = 0.2
    w6 = 0.1
    b1 = 0.5
    b2 = 0.5
    b3 = 1
    sample_data = get_sample()
    # Train
    for x in sample_data:
        w1, w2, w3, w4, w5, w6, loss = update_weights(get_predict(x[0], x[1], w1, w2, w3, w4, w5, w6,b1,b2,b3), x[2], x[0], x[1], w1,w2, w3, w4, w5, w6,b1,b2, 0.5)
        print("loss is ", loss)
    # Test
    print("final weight is, ", w1, w2, w3, w4, w5, w6)
    sample_test = get_sample()
    correctness = 0
    for x in sample_test:
        res = get_predict(x[0], x[1], w1, w2, w3, w4, w5, w6, b1, b2, b3)
        if res > 0.5:
            out = 1
        if res <= 0.5:
            out = 0
        if out == x[2]:
            correctness += 1

    print("accuracy is ", correctness / len(sample_test))
