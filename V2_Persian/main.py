#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import itertools
from game import Three
from player import Player


def run():
    results = {}.fromkeys((-1, 0, 1), 0)
    for i in range(300):
        board = Three()
        player_1 = Player(name='random', board=board)
        player_2 = Player(name='mcts', board=board, mcts=True)
        players_iter = itertools.cycle([player_1, player_2])
        while True:
            player = next(players_iter)
            action = player.choose_action()
            over, winner = board.update(action)
            # board.print_board()
            if over:
                results[winner] += 1
                break
            # time.sleep(0.5)
        if (i+1) % 5 == 1:
            print('result: ', results)


if __name__ == '__main__':
    run()

