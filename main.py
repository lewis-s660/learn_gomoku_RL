from datetime import datetime


from control import Control
from gomoku import Gomoku
from player_policy_gradient import PlayerPolicyGradient

size_board = (9, 9)
name = 'pattern_1'
path = 'weights'

environment = Gomoku(size_board)

player_black = PlayerPolicyGradient(True, size_board, path, name, mode='random')
player_white = PlayerPolicyGradient(False, size_board, path, name, mode='random2')

players = [player_black, player_white]

control = Control(environment, players)

for _ in range(1):
    experience = control.play(100)

    for i in range(len(players)):
        players[i].fit(experience[i], epochs=10000, size_batch=30)


