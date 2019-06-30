import Arena
from MCTS import MCTS

from games.tictactoe.TicTacToeGame import TicTacToeGame as Game
from games.tictactoe.mxnet.NNet import NNetWrapper as NNet
from utils import *
import numpy as np

MODEL_FOLDER = './trained_model/'
MODEL_NAME = 'padawan0.network'
BOARD_SIZE = 3
ARGS = dotdict({
        'numMCTSSims': 50,
        'cpuct':1.0,
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 1,
        'batch_size': 64,
        'cuda': False,
        'num_channels': 512,
    })

game = Game(BOARD_SIZE, False)
neural_network = NNet(game, args)
neural_network.load_checkpoint(MODEL_FOLDER,MODEL_NAME)
mcts = MCTS(game, neural_network, ARGS)

network = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

def display(board, valids, currPlayer, init_board, base_api, message=None):
    row,col = board.shape
    moves = []
    for i in range(len(valids)):
        if valids[i]:
            moves.append((int(i/row), int(i%col)))
    html = '<table border="1">'
    for r in range(0,row):
        html += '<tr>'
        for c in range(0,col):
            piece = board[r][c]
            if piece == -1: html += '<th>{}</th>'.format("X")
            elif piece == 1: html += '<th>{}</th>'.format("O")
            else:
                hfer = '----'
                if (r,c) in moves:
                    next_board = np.copy(board)
                    next_board[r][c] = -1 #Player Simbol
                    query = ','.join([str(e) for e in next_board.reshape(-1)]) + ',' + currPlayer
                    hfer = '<a href="{}?board={}">{}</a>'.format(base_api, query, hfer)
                html += '<th>{}</th>'.format(hfer)
        html += '</tr>'
    html += '</table>'
    html += '<br>{}<br><a href="{}?board={}">Start Over!</a>'.format(message, base_api, '{},{}'.format(init_board, currPlayer)) if message else ""
    return html

def playGame(board, base_api = 'localhost', init_board = '0,0,0,0,0,0,0,0,0'):
    if len(board) == 0:
        board = '{},N'.format(init_board)

    board = board.split(',')
    currPlayer = board[-1]
    board = np.array(board[:-1]).reshape(3,3).astype(int)

    status = game.getGameEnded(board, 1)
    if status != 0:
        message = "You Lose" if status == 1 else ("You Win" if status == -1 else "Draw.")
        currPlayer = "H" if currPlayer == "N" else "N"
        return display(board, [], currPlayer, init_board, base_api, message)

    action = network(game.getCanonicalForm(board, 1))
    valids = game.getValidMoves(game.getCanonicalForm(board, 1),1)

    if np.sum(board) == 0 and currPlayer == "H":
        return display(board, valids, currPlayer, init_board, base_api)

    board, _ = game.getNextState(board, 1, action)
    valids = game.getValidMoves(game.getCanonicalForm(board, 1),1)
    return display(board, valids, currPlayer, init_board, base_api)

def hello(event, context):
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """
