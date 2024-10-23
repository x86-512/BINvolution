# QCNN
This is a project that is written in Qiskit to make a classical-hybrid quantum convolutional neural network trained on a subset of the MNIST dataset.
This project was my attempt to write a Classical-Hybrid Quantum (CQ) Convolutional Neural Network.

The accuracy, training time, and running time are really bad (as expected) due to Barren Plateaus, unoptimized parameters, and suboptimal observables for the network.
I strongly recommend NOT running this on an actual quantum computer because the Decoherence will be TERRIBLE due to the circuit depth. Also this will suck up all your minutes(if you are on IBM's free plan).

# How it works:

Encoding Layer:
Each pixel in the image is mapped to a qubit. The qubit's phase rotated by its pixel's corresponding value in the flattened tensor. I am aware that the expressibility of this is not great, but this was my attempt at recreating the ZFeatureMap circuit from Qiskit.

More documentation coming soon
