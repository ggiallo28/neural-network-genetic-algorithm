import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .TicTacToeNNet import TicTacToeNNet as onnet

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

#args = dotdict({
#    'lr': 0.001,
#    'dropout': 0.3,
#    'epochs': 3,
#    'batch_size': 64,
#    'cuda': True,
#    'num_channels': 512,
#})

class NNetWrapper(NeuralNet):
    def __init__(self, game, args):
        self.nnet = onnet(game, args)
        self.game = game
        self.name = str(hex(id(self)))
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.loss = 99999999999
        self.args = args

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        end = time.time()
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        train_history = self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = self.args.batch_size, epochs = self.args.epochs, verbose=0)
        self.loss = train_history.history['loss']

        v0 = len(examples)
        v1 = round(time.time()-end,2)
        v2 = round(train_history.history['loss'][0],5)
        v3 = round(train_history.history['pi_loss'][0],5)
        v4 = round(train_history.history['v_loss'][0],5)
        print('Examples {} | Time Total: {}s | loss {} | pi_loss {} | v_loss {}'.format(v0,v1,v2,v3,v4))

    def predict(self, board):
        """
        board: np array with board
        """
        # timing

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        with self.nnet.session.as_default():
            with self.nnet.graph.as_default():
                pi, v = self.nnet.model.predict(board)

        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)
        return self

    def get_weights(self):
        return np.array(self.nnet.model.get_weights())

    def set_weights(self, weights):
        self.nnet.model.set_weights(weights)
        return self

    def get_loss(self):
        return self.loss
