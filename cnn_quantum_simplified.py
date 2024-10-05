import tensorflow as tf
import numpy as num
import matplotlib.pyplot as plt
import cudaq
from cudaq import spin
#from qiskit.circuit import QuantumCircuit

#def quantum_func(inputs):
#
#Start off with qiskit
#Get expectation of 2 qubits after pooling


#def quantum_layer(inputs):

    #Convolution
#    conv = QuantumCircuit(4)
#    conv.rz(-num.pi/2,1)
#    conv.cx(1,0)
#    conv.rz(inputs[0], 0)
#    conv.ry(inputs[1], 1)
#    conv.cx(0,1)
#    conv.rz(inputs[3], 0)
#    conv.ry(inputs[2], 1)
#    conv.cx(1,0)
#    conv.rz(num.pi/2,0)
#    conv.barrier()
    
#    conv.rz(-num.pi/2,3)
#    conv.cx(3,2)
#    conv.rz(inputs[4], 2)
#    conv.ry(inputs[5], 3)
#    conv.cx(2,3)
#    conv.rz(inputs[7], 2)
#    conv.ry(inputs[6], 3)
#    conv.cx(3,2)
#    conv.rz(num.pi/2,2)
#    conv.barrier()
    
#    conv.rz(-num.pi/2,2)
#    conv.cx(2,1)
#    conv.rz(inputs[8], 1)
#    conv.ry(inputs[9], 2)
#    conv.cx(1,2)
#    conv.rz(inputs[11], 1)
#    conv.ry(inputs[10], 2)
#    conv.cx(2,1)
#    conv.rz(num.pi/2,2)
#    conv.barrier()
        
#    conv.rz(-num.pi/2,3)
#    conv.cx(3,0)
#    conv.rz(inputs[12], 0)
#    conv.ry(inputs[13], 3)
#    conv.cx(0,3)
#    conv.rz(inputs[15], 0)
#    conv.ry(inputs[14], 3)
#    conv.cx(3,0)
#    conv.rz(num.pi/2,0)
#    conv.barrier()
    
    #Pooling layer
#    conv.rz(-num.pi/2,1)
#    conv.cx(1,0)
#    conv.rz(inputs[0], 0)
#    conv.ry(inputs[1], 1)
#    conv.cx(0,1)
#    conv.rz(inputs[3], 0)
#    conv.ry(inputs[2], 1)
#    conv.barrier()
    
#    conv.rz(-num.pi/2,3)
#    conv.cx(3,2)
#    conv.rz(inputs[4], 2)
#    conv.ry(inputs[5], 3)
#    conv.cx(2,3)
#    conv.rz(inputs[7], 2)
#    conv.ry(inputs[6], 3)
#    conv.barrier()
    
#    conv.rz(-num.pi/2,2)
#    conv.cx(2,1)
#    conv.rz(inputs[8], 1)
#    conv.ry(inputs[9], 2)
#    conv.cx(1,2)
#    conv.rz(inputs[11], 1)
#    conv.ry(inputs[10], 2)
#    conv.barrier()
        
#    conv.rz(-num.pi/2,3)
#    conv.cx(3,0)
#    conv.rz(inputs[12], 0)
#    conv.ry(inputs[13], 3)
#    conv.cx(0,3)
#    conv.rz(inputs[15], 0)
#    conv.ry(inputs[14], 3)

#    expectation_vals = [0,0]
#    return num.array(expectation_vals)


def quantum_layer(inputs):

    #Convolution
    qubits = cudaq.qvector(4)
    rz(-num.pi/2,qubits[1])
    cx(qubits[1],qubits[0])
    rz(inputs[0], qubits[0])
    ry(inputs[1], qubits[1])
    cx(qubits[0],qubits[1])
    rz(inputs[3], qubits[0])
    ry(inputs[2], qubits[1])
    cx(1,0)
    rz(num.pi/2,0)
    
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

    expectation_vals = [0,0]
    return num.array(expectation_vals)
    #Expectation value of [2,3], turn into np array
    #Pool to qubits 3 and 4, then return the expectation values of the 2 qubits and turn it into a tensor
    


class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, units=2, input_dim=2):
        super(QuantumLayer, self).__init__()
        self.units = units

        #self.circuit = 
        print("Settings Weights")


    def build(self, input_shape=(1,2)):
        
        self.weight1 = self.add_weight(shape=(1,2), initializer="random_normal", trainable = True) #Passed into ansatz
        self.weight2 = self.add_weight(shape=(1,2), initializer="random_normal", trainable = True) #Passed into ansatz
        pass

    def call(self, inputs):
        print("CALLING QLAYER")
        #flat_inputs = tf.reshape(inputs, [-1])
        print(inputs)

        if tf.executing_eagerly():
            print("EAGER")
            ansatz_inputs = inputs.numpy() #Tensorflow to numpy
            print(inputs.numpy())
            exp_vals = tf.convert_to_tensor(quantum_layer(inputs))
            print(exp_vals)
            return exp_vals
        print("NOT EAGER")
        return inputs
        #Flatten the tensor into 16
        #Convert the 2x2 to 4 straight, then determine which elements of the incoming feature mmap are mapped to which pixel
        #https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html Feature maps are mapped like this

        #
        #if tf.is_executing_eagerly():
        #    final_output = []
        #    for i in range(inputs.shape[0]):
        #        out = self.circuit(inputs[i].numpy())
        #        final_output.append(list(out))
        #returnable = self.qnn.forward(inputs,(self.weight1, self.weight2))
        #tensorflow quantum function
        #breakpoint()
        #return returnable
        #return inputs

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

    def setup(self):
        self.model.add(tf.keras.layers.Conv2D(32,(3,3), activation="relu", input_shape = (28,28,1)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Conv2D(16,(3,3), activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(tf.keras.layers.Conv2D(4,(4,4), activation="relu"))  #It is now at 2x2x4, QCNN Layer should have 4 qubits, pool it to 2, add an int. layer and map it to the output
        self.model.add(tf.keras.layers.Flatten())
        #self.model.add(tf.keras.layers.Dense(64,activation="relu"))
        
        #Alt:
        self.model.add(QuantumLayer(4,(2,2)))
        self.model.add(tf.keras.layers.Dense(2, activation="relu"))
        
        self.model.add(tf.keras.layers.Dense(2,activation="softmax")) #Just use qiskit at the end
        print(self.model.summary())
        #self.model.add(QuantumLayer())


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
