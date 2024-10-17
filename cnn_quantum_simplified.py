import tensorflow as tf
import numpy as num
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.circuit.library import ZGate

#Issues: Convert to Tensorflow tensor, expectation value
def quantum_layer(inputs):

    #Convolution
    conv = QuantumCircuit(4)
    conv.rz(-num.pi/2,1)
    conv.cx(1,0) #Connects all of the parameters, by entangling 2 qubits, you are allowing each qubit(pixel) to de
    conv.rz(inputs[0], 0)
    conv.ry(inputs[1], 1)
    conv.cx(0,1)
    conv.rz(inputs[3], 0)
    conv.ry(inputs[2], 1)
    conv.cx(1,0)
    conv.rz(num.pi/2,0)
    conv.barrier()
    
    conv.rz(-num.pi/2,3)
    conv.cx(3,2)
    conv.rz(inputs[4], 2)
    conv.ry(inputs[5], 3)
    conv.cx(2,3)
    conv.rz(inputs[7], 2)
    conv.ry(inputs[6], 3)
    conv.cx(3,2)
    conv.rz(num.pi/2,2)
    conv.barrier()
    
    conv.rz(-num.pi/2,2)
    conv.cx(2,1)
    conv.rz(inputs[8], 1)
    conv.ry(inputs[9], 2)
    conv.cx(1,2)
    conv.rz(inputs[11], 1)
    conv.ry(inputs[10], 2)
    conv.cx(2,1)
    conv.rz(num.pi/2,2)
    conv.barrier()
        
    conv.rz(-num.pi/2,3)
    conv.cx(3,0)
    conv.rz(inputs[12], 0)
    conv.ry(inputs[13], 3)
    conv.cx(0,3)
    conv.rz(inputs[15], 0)
    conv.ry(inputs[14], 3)
    conv.cx(3,0)
    conv.rz(num.pi/2,0)
    conv.barrier()
    
    #Pooling layer
    conv.rz(-num.pi/2,1)
    conv.cx(1,0)
    conv.rz(inputs[0], 0)
    conv.ry(inputs[1], 1)
    conv.cx(0,1)
    conv.rz(inputs[3], 0)
    conv.ry(inputs[2], 1)
    conv.barrier()
   
    conv.rz(-num.pi/2,3)
    conv.cx(3,2)
    conv.rz(inputs[4], 2)
    conv.ry(inputs[5], 3)
    conv.cx(2,3)
    conv.rz(inputs[7], 2)
    conv.ry(inputs[6], 3)
    conv.barrier()
    
    conv.rz(-num.pi/2,2)
    conv.cx(2,1)
    conv.rz(inputs[8], 1)
    conv.ry(inputs[9], 2)
    conv.cx(1,2)
    conv.rz(inputs[11], 1)
    conv.ry(inputs[10], 2)
    conv.barrier()
        
    conv.rz(-num.pi/2,3)
    conv.cx(3,0)
    conv.rz(inputs[12], 0)
    conv.ry(inputs[13], 3)
    conv.cx(0,3)
    conv.rz(inputs[15], 0)
    conv.ry(inputs[14], 3)

    est = Estimator()
    res = est.run([conv],[ZGate().to_matrix()]).result()

    expectation_vals = [res.values[0],res.values[1]]
    return num.array(expectation_vals)

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, units=2, input_dim=2):
        super(QuantumLayer, self).__init__()
        self.units = units


    def build(self, input_shape=(1,2)):
        
        self.weight1 = self.add_weight(shape=(1,2), initializer="random_normal", trainable = True) #Passed into ansatz
        self.weight2 = self.add_weight(shape=(1,2), initializer="random_normal", trainable = True) #Passed into ansatz
        pass

    def call(self, inputs):

        if tf.executing_eagerly(): #Necessary to get the tensors to numpy to work to run the quantum layer
            ansatz_inputs = inputs.numpy()
            exp_vals = tf.convert_to_tensor(quantum_layer(ansatz_inputs))
            return tf.convert_to_tensor(exp_vals)
        return inputs
        #Flatten the tensor into 16
        #Convert the 2x2 to 4 straight, then determine which elements of the incoming feature mmap are mapped to which pixel
        #https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html Feature maps are mapped like this


class Conv_NN():
    def __init__(self):
        self.model = tf.keras.models.Sequential()
        mnist = tf.keras.datasets.mnist
        (self.train_img, self.train_lbl), (self.test_img, self.test_lbl) = mnist.load_data()

    def filter(self):
        train_filter = num.where((self.train_lbl==0)|(self.train_lbl==1))
        test_filter = num.where((self.test_lbl==0)|(self.test_lbl==1))
        self.train_img, self.train_lbl = self.train_img[train_filter], self.train_lbl[train_filter]
        self.test_img, self.test_lbl = self.test_img[test_filter], self.test_lbl[test_filter]

    #Add an option for eager execution
    def setup(self):
        self.model.add(tf.keras.layers.Conv2D(32,(3,3), activation="relu", input_shape = (28,28,1)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Conv2D(16,(3,3), activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Conv2D(4,(4,4), activation="relu"))  #It is now at 2x2x4, QCNN Layer should have 4 qubits, pool it to 2, add an int. layer and map it to the output
        self.model.add(tf.keras.layers.Flatten()) #Quantum layers need to be in a flat tensor
        
        #Alt:
        self.model.add(QuantumLayer(4,(2,2)))
        self.model.add(tf.keras.layers.Dense(2, activation="relu"))
        
        self.model.add(tf.keras.layers.Dense(2,activation="softmax")) #Just use qiskit at the end
        #print(self.model.summary())


    def compile(self):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=["accuracy"])

    def train(self):
        self.model.fit(self.train_img, self.train_lbl, epochs=5, batch_size=128)
        self.model.evaluate(self.test_img, self.test_lbl, batch_size=128, verbose=2)

def main():
    nn = Conv_NN()
    nn.filter()
    nn.setup()
    nn.compile()
    nn.train()

if __name__=="__main__":
    main()
