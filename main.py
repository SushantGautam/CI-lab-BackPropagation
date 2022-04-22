import matplotlib.pyplot as plt
import numpy as np

number_of_hidden_units = 10
learning_rate = 0.075
num_iterations = 3000
regression = False

Test_Accuracy_label = "Test Accuracy"

# Load Train Data

data = np.genfromtxt('IRIS.DAT',
                     dtype=(float, float, float, float, int, int, int),
                     )
data = np.array([list(ele) for ele in data])
X = data[:, [0, 1, 2, 3]].T  # input
Y = data[:, [4, 5, 6]].T  # output

# Load Lest Data
data_test = np.genfromtxt('IRIS_TEST.DAT',
                          dtype=(float, float, float, float, int, int, int),
                          )
data_test = np.array([list(ele) for ele in data_test])
Xt = data_test[:, [0, 1, 2, 3]].T  # input
Yt = data_test[:, [4, 5, 6]].T  # output

# 1. Deciding the shapes of Weight and bias matrix

# X --> input dataset of shape (input size, number of examples)
# Y --> labels of shape (output size, number of examples)
# Randomly Initialize
np.random.seed(0)
W1 = np.random.randn(number_of_hidden_units, X.shape[0])
b1 = np.zeros(shape=(number_of_hidden_units, 1))

W2 = np.random.randn(Y.shape[0], number_of_hidden_units)
b2 = np.zeros(shape=(Y.shape[0], 1))


# 2. Initializing sigmoid function to be used

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


# 3. Implementing the forward propagation method
def forward_prop(X, W1, W2, b1, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # here the cache is the data of previous iteration
    # This will be used for backpropagation
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


# 4. Implementing the cost calculation
# Here Y is actual output
def compute_cost(A2, Y):
    global regression
    m = Y.shape[1]
    if regression:
        cost = np.sum(((A2 - Y)) ** 2) / m
        pass
    else:
        cost_sum = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
        cost = - np.sum(cost_sum) / m
    # Squeezing to avoid unnecessary dimensions
    cost = np.squeeze(cost)
    return cost


def Evaluate(X, W1, W2, b1, b2):
    global regression

    Y, cache = np.array(forward_prop(X, W1, W2, b1, b2))
    if regression:
        return np.average(((Y.T - Yt.T) / Y.T) ** 2)
    else:
        Yindex = np.argmax(Y.T, axis=1)
        Ytindex = np.argmax(Yt.T, axis=1)  # Yt is global
        difference = [True if ((ey - eyt) == 0) else False for ey, eyt in zip(Yindex, Ytindex)]
        return np.average(difference)


# 5. Backpropagation and optimizing

def back_propagate(W1, b1, W2, b2, cache):
    # Retrieve also A1 and A2 from dictionary "cache"
    A1 = cache['A1']
    A2 = cache['A2']

    m = Y.shape[1]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    # Updating the parameters according to algorithm
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, W2, b1, b2


# 6. prediction and visualizing the output

# Here num_iteration is epochs

logVar = []
for i in range(0, num_iterations):

    # Forward propagation. Inputs: "X, parameters". return: "A2, cache".
    A2, cache = forward_prop(X, W1, W2, b1, b2)

    # Cost function. Inputs: "A2, Y". Outputs: "cost".
    cost = compute_cost(A2, Y)

    # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
    W1, W2, b1, b2 = back_propagate(W1, b1, W2, b2, cache)

    # Acuuracy function. Inputs: "A2, Y". Outputs: "cost".
    accuracy = Evaluate(Xt, W1, W2, b1, b2)

    # Print the cost every 100 iterations
    if i % 100 == 0:
        print("Iteration % i ->    Cost: % f    %s: % f" % (i, cost, Test_Accuracy_label, accuracy))
        logVar.append([i, cost, accuracy])

# red dashes, blue squares and green triangles
plt.plot([i[0] for i in logVar], [i[1] for i in logVar], 'r--', label='Cost')
plt.plot([i[0] for i in logVar], [i[2] for i in logVar], 'b--', label=Test_Accuracy_label)
plt.xlabel('Iterations -->', fontsize=18)
plt.ylabel('Performance -->', fontsize=16)
plt.title("Training Result\n\n" + "hidden units: " + str(number_of_hidden_units) +
          "   lr rate: " + str(learning_rate) +
          "   iter: " + str(num_iterations) +
          "   dataset: " + "IRIS" + "\n")
plt.legend()
plt.ylim(top=1.1)
plt.show()
