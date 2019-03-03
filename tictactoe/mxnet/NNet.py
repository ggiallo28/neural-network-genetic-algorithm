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

from mxnet import nd, gpu, gluon, init, autograd
import argparse
from .TicTacToeNNet import TicTacToeNNet as onnet

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 1,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.loss = 0

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)[np.newaxis, :, :]
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        dataset_train = gluon.data.dataset.ArrayDataset(input_boards, target_pis, target_vs)
        train_data = gluon.data.DataLoader(dataset_train,batch_size=args.epochs,shuffle=True,num_workers=4)

        for epoch in range(args.epochs):
            for input_board, target_pi, target_v in train_data:
                print(epoch)
                if args.cuda:
                    input_board = input_board.as_in_context(ctx)
                    target_pi = target_pi.as_in_context(ctx)
                    target_v = target_v.as_in_context(ctx)
                with autograd.record():
                    pi, v = self.nnet.predict(input_board)
                    self.pi_loss = self.nnet.pi_loss(pi,target_pis)
                    self.v_loss = self.nnet.v_loss(out,target_vs)
                    self.loss = self.pi_loss + self.v_loss
                self.loss.backward()
                self.nnet.trainer.step(args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = board[np.newaxis, np.newaxis, :, :]

        # run
        pi, v = self.nnet.predict(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)

    def get_weights(self):
        return self.nnet.model.get_weights()

    def set_weights(self, weights):
        self.nnet.model.set_weights(weights)

    def get_loss(self):
        return self.loss
