import time
import numpy as np
import matplotlib.pyplot as plt

def segmenta_dataset(x, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    n_samples = x.shape[0]
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype=int)
    if num_classes is None:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    for i in range(n):
        categorical[i, y[i]] = 1
    return categorical

def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred))

class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.input = None
        self.output = None
        
    def forward(self, X):
        self.input = X
        z = np.dot(X, self.weights) + self.biases
        
        if self.activation == 'relu':
            self.output = relu(z)
        elif self.activation == 'softmax':
            self.output = softmax(z)
        else:
            self.output = z
            
        return self.output
    
    def backward(self, d_output, learning_rate):
        if self.activation == 'relu':
            d_output = d_output * deriv_relu(self.output)
            
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)
        d_input = np.dot(d_output, self.weights.T)
        
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return d_input

class SequentialModel:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
    
    def fit(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None, learning_rate=0.01):
        X_val, y_val = validation_data if validation_data else (None, None)
        history = {'loss': [], 'val_loss': []}
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                output = X_batch
                for layer in self.layers:
                    output = layer.forward(output)
                
                loss = categorical_crossentropy(y_batch, output)
                epoch_loss += loss * X_batch.shape[0]
                
                d_output = (output - y_batch) / y_batch.shape[0]
                for layer in reversed(self.layers):
                    d_output = layer.backward(d_output, learning_rate)
            
            epoch_loss /= n_samples
            history['loss'].append(epoch_loss)
            
            if X_val is not None:
                val_output = self.predict(X_val)
                val_loss = categorical_crossentropy(y_val, val_output)
                history['val_loss'].append(val_loss)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
                
        return history
    
    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

data = np.genfromtxt("treino_sinais_vitais_com_label.txt", delimiter=',', skip_header=1)
X = data[:, 1:-1]
Y = data[:, -1].reshape(-1, 1)

X = np.delete(X, 5, axis=1)

X_train, X_test, Y_train, Y_test = segmenta_dataset(X, Y, test_size=0.2, random_state=50)

Y_train_cat = to_categorical(Y_train - 1, num_classes=4)
Y_test_cat = to_categorical(Y_test - 1, num_classes=4)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)

start_time = time.time()
model = SequentialModel()
model.add(DenseLayer(input_size=X_train.shape[1], output_size=64, activation='relu'))
model.add(DenseLayer(input_size=64, output_size=32, activation='relu'))
model.add(DenseLayer(input_size=32, output_size=4, activation='softmax'))

history = model.fit(X_train, Y_train_cat, 
                   epochs=1500, 
                   batch_size=4, 
                   validation_data=(X_test, Y_test_cat),
                   learning_rate=0.008)

print(f"Time Spent Training: {time.time() - start_time}s")

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1) + 1
true_classes = Y_test.flatten()
accuracy = np.mean(predicted_classes == true_classes)
print(f"Accuracy on entire dataset: {accuracy * 100:.2f}%")


"""
plt.plot(history['loss'], label='Loss (Training)')
plt.plot(history['val_loss'], label='Loss (Validation)')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
"""