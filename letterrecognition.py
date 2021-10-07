import numpy as np
import requests, gzip, os, hashlib
from Dataset import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




'''
#fetch data
path=os.getcwd() + '/data'
def fetch(url):
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
'''

dataset = Letters()
x=[]
y=[]
for i in dataset.images:
    x.append(i)
    y.append(i.target)
    for b in blur(i):
        pass
        x.append(b)
        y.append(i.target)



#X_train, X_test, Y_train, Y_test = train_test_split(np.array(x), np.array(y), test_size=0.5, shuffle=True)
X_train = np.asarray(x)
X_test = np.asarray(x)
Y_train = np.squeeze(y)
Y_test = np.squeeze(y)


'''
#Validation split
rand=np.arange(60000)
np.random.shuffle(rand)
train_no=rand[:50000]

val_no=np.setdiff1d(rand,train_no)

X_train,X_val=X[train_no,:,:],X[val_no,:,:]
Y_train,Y_val=Y[train_no],Y[val_no]
'''


def init(x,y):
    layer=np.random.uniform(-1.,1.,size=(x,y))/np.sqrt(x*y)
    return layer.astype(np.float32)

np.random.seed(42)
NN1 = init(5*5,3)


#Sigmoid funstion
def sigmoid(x):
    return 1/(np.exp(-x)+1)

#derivative of sigmoid
def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)


#Softmax
def softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)

#derivative of softmax
def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))


# forward and backward pass
def forward_backward_pass(x, y):
    targets = np.zeros((len(y), 3), np.float32)
    targets[range(targets.shape[0]), y] = 1

    x_nn1 = x.dot(NN1)
    x_sigmoid = sigmoid(x_nn1)

    out = softmax(x_sigmoid)

    error = 2 * (out - targets) / out.shape[0] * d_softmax(x_sigmoid)
    update_nn1 = x.T @ error

    return out, update_nn1


def plot9(images):
    fig, ax1 = plt.subplots(3, 3)
    fig.set_facecolor('g')
    for i, img in enumerate(images):
        res = sigmoid(img.reshape(-1, 5 * 5).dot(NN1))
        ax1[i % 3][int(np.floor(i/3))].axis('off')
        ax1[i % 3][int(np.floor(i/3))].set_title(str(letter(np.argmax(res))) + ': conf ' + str(round(np.max(softmax(res.reshape(3,-1))),2)))
        ax1[i % 3][int(np.floor(i/3))].imshow(img, cmap='gray')
    plt.show()

epochs = 10000
lr = 0.001
batch = 9
losses, accuracies, val_accuracies = [], [], []

for i in range(epochs):
    sample = np.random.randint(0, X_train.shape[0], size=(batch))
    X = X_train[sample].reshape((-1, 5 * 5))
    Y = Y_train[sample]

    out, update_nn1 = forward_backward_pass(X, Y)

    category = np.argmax(out, axis=1)
    accuracy = (category == Y).mean()
    accuracies.append(accuracy)

    loss = ((category - Y) ** 2).mean()
    losses.append(loss.item())

    NN1 = NN1 - lr * update_nn1

    if (i % 20 == 0):
        X_test = X_test.reshape((-1, 5 * 5))
        val_out = np.argmax(softmax(sigmoid(X_test.dot(NN1))), axis=1)
        val_acc = (val_out == Y_test).mean()
        val_accuracies.append(val_acc.item())
    if (i % 500 == 0): print(f'For {i}th epoch: train accuracy: {accuracy:.3f} | validation accuracy:{val_acc:.3f}')


plot9(x)

predictions = np.asarray([letter(np.argmax(sigmoid(img.reshape(-1, 5 * 5).dot(NN1)))) for img in dataset.images])
labels = np.asarray([letter(img.target) for img in dataset.images])

disp = ConfusionMatrixDisplay.from_predictions(labels,predictions)
disp.ax_.set_title('Confusion Matrix')
disp.plot()
