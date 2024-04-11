import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax


def weights_generator():
    init_range = 0.01
    weights = np.random.uniform(-init_range, init_range, (2, 1))
    biases = np.random.uniform(-init_range, init_range, 1)
    learning_rate = 0.02
    # return weights, biases, learning_rate
    return weights, biases, learning_rate


def training(weights, biases, learning_rate, inputs, targets, obs):
    global deltas, loss
    for i in range(100):
        outputs = np.dot(inputs, weights) + biases
        deltas = outputs - targets
        loss = np.sum(deltas ** 2) / 2 / obs
        print(loss)
        deltas_scaled=deltas/obs
        weights=weights-learning_rate * np.dot(inputs.T, deltas_scaled)
        biases=biases-learning_rate*np.sum(deltas_scaled)


    print(weights, biases)
    return outputs



def main():
    obs = 10000
    xs = np.random.uniform(low=-10, high=10, size=(obs, 1))
    zx = np.random.uniform(low=-10, high=10, size=(obs, 1))
    inputs = np.column_stack((xs, zx))
    print(inputs.shape)
    noise = np.random.uniform(-1, 1, (obs, 1))
    targets = 2 * xs - 3 * zx + 5 + noise
    targets = targets.reshape(obs, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, zx, targets, c='r')

    ax.set_xlabel('xs')
    ax.set_ylabel('zx')
    ax.set_zlabel('Targets')
    ax.view_init(azim=100)
    plt.show()
    targets = targets.reshape(obs, 1)
    # fetch weights, biases and learning rate
    weights, biases, learning_rate = weights_generator()
    output=training(weights, biases, learning_rate, inputs, targets, obs)
    plt.plot(output,targets)
    plt.xlabel('outputs')
    plt.ylabel('targets')
    plt.show()





if __name__ == '__main__':
    main()
