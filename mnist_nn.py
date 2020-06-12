import torch as T
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 

def encode_one_hot(y, num_labels=10):
    v = T.zeros(num_labels, y.shape[0])
    for i, val in enumerate(y):
        v[val,i] = 1.0 
    return v 

def add_bias(layer, row):
    if row:
        newlayer = T.ones((layer.shape[0]+1, layer.shape[1]))
        newlayer[1:, :] = layer 
    else:
        newlayer = T.ones((layer.shape[0], layer.shape[1] + 1))
        newlayer[:,1:] = layer 
    return newlayer 

def init_weight(n_input, n_hidden, n_output, batch_size):
    n_hidden1, n_hidden2 = n_hidden
    w1 = T.randn((n_hidden1, n_input+1), dtype=T.float)
    w2 = T.randn((n_hidden2, n_hidden1+1), dtype=T.float)
    w3 = T.randn((n_output, n_hidden2+1), dtype=T.float)
    return w1,w2,w3 

def forward(input, w1, w2, w3):
    a1 = T.reshape(input, shape=(input.shape[0], -1))
    a1 = add_bias(a1, False)

    z2 = w1.matmul(T.transpose(a1,0,1))
    a2 = T.sigmoid(z2)
    a2 = add_bias(a2, True)

    z3 = w2.matmul(a2)
    a3 = T.sigmoid(z3)
    a3 = add_bias(a3, True)

    z4 = w3.matmul(a3)
    a4 = T.sigmoid(z4)

    return a1, z2, a2, z3, a3, z4, a4 

def predict(a4):
    prediction = T.argmax(a4, dim=0)
    return prediction 

def compute_loss(prediction, label):
    a = -1*label*T.log(prediction)
    b = (1-label)*T.log(1-prediction)
    loss = T.sum(a - b)
    return loss

def backward(weights, outputs, label):
    w1, w2, w3 = weights 
    a1, z2, a2, z3, a3, z4, a4 = outputs 

    d4 = a4-label 
    d3 = T.transpose(w3[:,1:], 0, 1).matmul(d4) * T.sigmoid(z3)*(1-T.sigmoid(z3))
    d2 = w2[:,1:].matmul(d3)*T.sigmoid(z2)*(1-T.sigmoid(z2))
    grad1 = d2.matmul(a1)
    grad2 = d3.matmul(T.transpose(a2,0,1))
    grad3 = d4.matmul(T.transpose(a3,0,1))
    return grad1, grad2, grad3 

def gen_data(train_batch_size, test_batch_size=10):
    mnist_train = MNIST('mnist', train=True, download=True, transform=ToTensor())
    train_data_load = T.utils.data.DataLoader(mnist_train, batch_size=train_batch_size, shuffle=True, num_workers=8)
    mnist_test = MNIST('mnist', train=False, download=True, transform=ToTensor())
    test_data_load = T.utils.data.DataLoader(mnist_test, batch_size=test_batch_size, shuffle=True, num_workers=8)
    return train_data_load, test_data_load

if __name__ == "__main__":
    batch_size = 50
    n_input = 28*28

    n_hidden_1, n_hidden_2, n_output = 100, 100, 10
    w1, w2, w3 = init_weight(n_input, (n_hidden_1, n_hidden_2), n_output, batch_size)
    e = 0.001
    a = 0.001
    epochs = 20

    d1prev = T.zeros(w1.shape)
    d2prev = T.zeros(w2.shape)
    d3prev = T.zeros(w3.shape)

    train_loss = []
    train_acc = []
    train_data, test_data = gen_data(batch_size)

    for i in range(epochs):
        for j, (input, label) in enumerate(train_data):
            one_hot = encode_one_hot(label, num_labels=10)
            a1, z2, a2, z3, a3, z4, a4 = forward(input, w1, w2, w3)
            loss = compute_loss(a4, one_hot.float())
            grad1, grad2, grad3 = backward((w1, w2, w3), (a1, z2, a2, z3, a3, z4, a4), one_hot.float())
            d1, d2, d3 = e*grad1, e*grad2, e*grad3
            w1 -= d1 + d1prev*a
            w2 -= d2 + d2prev*a 
            w3 -= d3 + d3prev*a
            d1prev, d2prev, d3prev = d1, d2, d3
            train_loss.append(loss)
            predictions = predict(a4)
            wrong = T.where(predictions != label,
                            T.tensor([1.]), T.tensor([0.]))
            accuracy = 1 - T.sum(wrong)/batch_size 
            train_acc.append(accuracy.float())
        print('epoch', i, 'training accuracy %.2f' % T.mean(T.tensor(train_acc)).item())
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)
    ax.plot(train_loss, color='red')
    ax.set_xlabel('iterations')
    ax.set_ylabel('loss', color='red')
    ax.tick_params(axis='y', colors="red")
    ax2.plot(train_acc, color='blue')
    ax2.yaxis.tick_right()
    ax2.set_ylabel('accuracy', color='blue')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="blue")
    ax2.set_xticklabels([])
    plt.show()
    for (image, label) in test_data:
        image = image[0]
        label = label[0]
        fig, ax = plt.subplots(1)
        ax.imshow(T.reshape(image[0], shape=(28,28)), interpolation='nearest')
        a1, z2, a2, z3, a3, z4, a4 = forward(image, w1, w2, w3)
        prediction = predict(a4)
        ax.text(5, 5, 'Predicted: %i, Actual: %i' % (prediction, label.float()), bbox={'facecolor': 'white', 'pad': 10})
        plt.show()
        break



    print('\n-------------\n')
    print('EVALUATE TEST DATA\n')

    test_acc = []
    for j, (input, label) in enumerate(test_data):
        one_hot_label = encode_one_hot(label, num_labels=10)
        a1, z2, a2, z3, a3, z4, a4 = forward(input,w1,w2,w3)
        loss = compute_loss(a4, one_hot_label.float())

        predictions = predict(a4)
        wrong = T.where(predictions != label, T.tensor([1.]), T.tensor([0.]))
        accuracy = 1 - T.sum(wrong)/batch_size

        test_acc.append(accuracy)

    print('Testing Accuracy %.2f' % T.mean(T.tensor(test_acc)).item())
    