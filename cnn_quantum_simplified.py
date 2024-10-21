import tensorflow as tf
import numpy as num
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.circuit.library import ZGate

from qiskit.quantum_info import SparsePauliOp

#Issues: Convert to Tensorflow tensor, expectation value
    #Put inputs into a parametervector and 
    #DO not use ZFeatureMap, manually map the inputs

    #Convolution

BATCH_SIZE=128

def encoder(inputs):
    #print(inputs.tolist())

    #print(len(inputs.tolist()))
    #breakpoint()
    enc = QuantumCircuit(4)
    #enc.h([0,1,2,3])
    enc.h(0)
    enc.h(1)
    enc.h(2)
    enc.h(3)
    enc.p(inputs[0],0)
    enc.p(inputs[1],1)
    enc.p(inputs[2],2)
    enc.p(inputs[3],3)
    enc.h(0)
    enc.h(1)
    enc.h(2)
    enc.h(3)
    enc.p(inputs[4],0)
    enc.p(inputs[5],1)
    enc.p(inputs[6],2)
    enc.p(inputs[7],3)
    enc.h(0)
    enc.h(1)
    enc.h(2)
    enc.h(3)
    enc.p(inputs[8],0)
    enc.p(inputs[9],1)
    enc.p(inputs[10],2)
    enc.p(inputs[11],3)
    enc.h(0)
    enc.h(1)
    enc.h(2)
    enc.h(3)
    enc.p(inputs[12],0)
    enc.p(inputs[13],1)
    enc.p(inputs[14],2)
    enc.p(inputs[15],3)
    return enc

def convolution(kernel_weights): #Get weights as a tensor
    #print(kernel_weights)
    #print(len(kernel_weights))
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
    #
    ins = inputs
    wc = weights_conv
    wp = weights_pool
    ql = QuantumCircuit(4)
    ql.compose(encoder(ins), list(range(0,4)), inplace=True) #Inplace increases time
    ql.compose(convolution(ins), list(range(0,4)), inplace=True)
    ql.compose(pool(ins), list(range(0,4)), inplace=True)
    est = Estimator()
    observable_2 = SparsePauliOp.from_list([("IIXX", 1)]) #These cause HUGE loss, I'm pretty sure, I did trial and error
    observable_3 = SparsePauliOp.from_list([("IIXX", 1)])
    job_1 = est.run(ql, observable_2) 
    res_1 = job_1.result()
    job_2 = est.run(ql, observable_3)
    res_2 = job_2.result()
    expectation_vals = [res_1.values[0],res_2.values[0]]
    #print([res_1.values[0],res_2.values[0]])
    return num.array(expectation_vals)
    #return inputs

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, units=2, input_dim=2):
        super(QuantumLayer, self).__init__()
        self.units = units


    def build(self, input_shape=(1,4)):
        self.weight1 = self.add_weight(shape=(1,16), initializer="random_normal", trainable = True) #Passed into ansatz
        self.weight2 = self.add_weight(shape=(1,16), initializer="random_normal", trainable = True) #16 weights + pool

    def call(self, inputs):
        #print(len([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
        #quantum_layer([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        if tf.executing_eagerly():
            #print(inputs.shape)
            #breakpoint()
            ansatz_inputs = inputs.numpy()
            conv_weights = self.weight1.numpy()
            pool_weights = self.weight2.numpy()
            #Inputs and weights are batched
            #print(conv_weights[1])
            #print(len(conv_weights))
            out_list = []
            for i in range(len(inputs)): #Batch size
                exp_vals = quantum_layer(ansatz_inputs[i], conv_weights, pool_weights)
                #print(exp_vals)
                out_list.append(exp_vals)
                #print(out_list)
            #out_list = quantum_layer(ansatz_inputs, conv_weights, pool_weights)
            out_array = num.array(out_list)
            #print(out_array)
            #print(len(out_array))
            #print(tf.convert_to_tensor(out_array).shape)
            return tf.convert_to_tensor(out_array) 
        #128, 16
        return tf.convert_to_tensor(num.zeros((BATCH_SIZE,2))) #Tensors need the same output size
        #return tf.convert_to_tensor([0,0]) #you better be executing eagerly
        
        #Flatten the tensor into 16
        #Convert the 2x2 to 4 straight, then determine which elements of the incoming feature mmap are mapped to which pixel
        #https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html Feature maps are mapped like this


class Conv_NN():
    def __init__(self):
        self.model = tf.keras.models.Sequential()
        mnist = tf.keras.datasets.mnist
        (self.train_img, self.train_lbl), (self.test_img, self.test_lbl) = mnist.load_data()
        #tf.enable_eager_execution()

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
        
        #Alt:
        #self.model.add(QuantumLayer(4,(2,2)))
        self.model.add(QuantumLayer(4,2))
        self.model.add(tf.keras.layers.Dense(2, activation="relu"))
        
        self.model.add(tf.keras.layers.Dense(2,activation="softmax"))
        #print(self.model.summary())


    def compile(self):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=["accuracy"])

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
