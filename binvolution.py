import tensorflow as tf
import numpy as num
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.circuit.library import ZGate

from qiskit.quantum_info import SparsePauliOp

BATCH_SIZE=128 #May downgrade to 64

def encoder(inputs):
    enc = QuantumCircuit(4)
    for i in range(4):
        enc.h(i)
        enc.p(inputs[i],i)
        enc.h(i)
        enc.p(inputs[i+4],i)
        enc.h(i)
        enc.p(inputs[i+8],i)
        enc.h(i)
        enc.p(inputs[i+12],i)
    return enc

def convolution(kernel_weights): #Get weights as a tensor
    conv = QuantumCircuit(4)
    #This circuit uses a kernel of length 2, so 2 qubits are operated on at a time
    conv.rz(-num.pi/2,1)
    conv.cx(1,0) #Connects all of the parameters, by entangling 2 qubits, you are allowing each qubit(pixel) to de
    conv.rz(kernel_weights[0], 0)
    conv.ry(kernel_weights[1], 1)
    conv.cx(0,1)
    conv.rz(kernel_weights[3], 0)
    conv.ry(kernel_weights[2], 1)
    conv.cx(1,0)
    conv.rz(num.pi/2,0)
    conv.barrier()
    
    conv.rz(-num.pi/2,3)
    conv.cx(3,2)
    conv.rz(kernel_weights[4], 2)
    conv.ry(kernel_weights[5], 3)
    conv.cx(2,3)
    conv.rz(kernel_weights[7], 2)
    conv.ry(kernel_weights[6], 3)
    conv.cx(3,2)
    conv.rz(num.pi/2,2)
    conv.barrier()
    
    conv.rz(-num.pi/2,2)
    conv.cx(2,1)
    conv.rz(kernel_weights[8], 1)
    conv.ry(kernel_weights[9], 2)
    conv.cx(1,2)
    conv.rz(kernel_weights[11], 1)
    conv.ry(kernel_weights[10], 2)
    conv.cx(2,1)
    conv.rz(num.pi/2,2)
    conv.barrier()
        
    conv.rz(-num.pi/2,3)
    conv.cx(3,0)
    conv.rz(kernel_weights[12], 0)
    conv.ry(kernel_weights[13], 3)
    conv.cx(0,3)
    conv.rz(kernel_weights[15], 0)
    conv.ry(kernel_weights[14], 3)
    conv.cx(3,0)
    conv.rz(num.pi/2,0)
    return conv
    
    #Pooling layer
def pool(pool_weights):
    pool = QuantumCircuit(4)
    pool.rz(-num.pi/2,1)
    pool.cx(1,0)
    pool.rz(pool_weights[0], 0)
    pool.ry(pool_weights[1], 1)
    pool.cx(0,1)
    pool.rz(pool_weights[3], 0)
    pool.ry(pool_weights[2], 1)
    pool.barrier()
   
    pool.rz(-num.pi/2,3)
    pool.cx(3,2)
    pool.rz(pool_weights[4], 2)
    pool.ry(pool_weights[5], 3)
    pool.cx(2,3)
    pool.rz(pool_weights[7], 2)
    pool.ry(pool_weights[6], 3)
    pool.barrier()

    pool.rz(-num.pi/2,2)
    pool.cx(2,1)
    pool.rz(pool_weights[8], 1)
    pool.ry(pool_weights[9], 2)
    pool.cx(1,2)
    pool.rz(pool_weights[11], 1)
    pool.ry(pool_weights[10], 2)
    pool.barrier()

    pool.rz(-num.pi/2,3)
    pool.cx(3,0)
    pool.rz(pool_weights[12], 0)
    pool.ry(pool_weights[13], 3)
    pool.cx(0,3)
    pool.rz(pool_weights[15], 0)
    pool.ry(pool_weights[14], 3)
    return pool

def quantum_layer(inputs, weights_conv, weights_pool):
    ins = inputs
    wc = weights_conv
    wp = weights_pool
    ql = QuantumCircuit(4)
    ql.compose(encoder(ins), list(range(0,4)), inplace=True) #Inplace increases time
    ql.compose(convolution(ins), list(range(0,4)), inplace=True)
    ql.compose(pool(ins), list(range(0,4)), inplace=True)
    est = Estimator()
    observable_2 = SparsePauliOp.from_list([("IIXX", 1)]) #These cause LOTS of loss
    observable_3 = SparsePauliOp.from_list([("IIXX", 1)])
    job_1 = est.run(ql, observable_2) 
    res_1 = job_1.result()
    job_2 = est.run(ql, observable_3)
    res_2 = job_2.result()
    expectation_vals = [res_1.values[0],res_2.values[0]]
    return num.array(expectation_vals)

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, units=2, input_dim=2):
        super(QuantumLayer, self).__init__()
        self.units = units


    def build(self, input_shape=(1,4)):
        self.weight1 = self.add_weight(shape=(1,16), initializer="random_normal", trainable = True) #Passed into ansatz
        self.weight2 = self.add_weight(shape=(1,16), initializer="random_normal", trainable = True) #16 weights + pool

    def call(self, inputs):
        if tf.executing_eagerly():
            ansatz_inputs = inputs.numpy()
            conv_weights = self.weight1.numpy()
            pool_weights = self.weight2.numpy()
            out_list = []
            for i in range(len(inputs)): #Batch size
                exp_vals = quantum_layer(ansatz_inputs[i], conv_weights, pool_weights)
                out_list.append(exp_vals)
            out_array = num.array(out_list)
            return tf.convert_to_tensor(out_array)
        return tf.convert_to_tensor(num.zeros((BATCH_SIZE,2))) #Same output size regardlesss of return


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
        
        self.model.add(QuantumLayer(4,2))
        self.model.add(tf.keras.layers.Dense(2, activation="relu"))
        
        self.model.add(tf.keras.layers.Dense(2,activation="softmax"))
        #print(self.model.summary())

    def compile(self):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=["accuracy"]) #Backpropogation slows down due to barren plateaus, it barely increases throughout the 1st epoch

    def train(self):
        self.model.fit(self.train_img, self.train_lbl, epochs=5, batch_size=BATCH_SIZE)
        self.model.evaluate(self.test_img, self.test_lbl, batch_size=BATCH_SIZE, verbose=2)

def main():
    tf.config.run_functions_eagerly(True)
    nn = Conv_NN()
    nn.filter()
    nn.setup()
    nn.compile()
    nn.train()

if __name__=="__main__":
    main()
