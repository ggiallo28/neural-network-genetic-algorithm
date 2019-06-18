from othello.OthelloGame import OthelloGame as Game
from othello.tensorflow.NNet import NNetWrapper as nn
import tensorflow as tf
import numpy as np

male = nn(Game()).get_weights()
female = nn(Game()).get_weights()

baby = nn(Game())
weights = 4*(male * 0.5 + female * 0.5)
variables = baby.nnet.get_parameters()

#print(weights[0].shape)
#print(variables[0])
#print(dir(variables[0]))

sess = tf.Session(graph=baby.nnet.graph)

with sess.as_default():
    for w,v in zip(weights, variables):
        print(v)
        v.load(w, sess)


variables = baby.nnet.get_parameters()
for w,v in zip(weights, variables):
    with sess.as_default():
        print(v)
        print(np.sum(np.subtract(w,v.eval())))
