#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import itertools
import time

from player import Player
from game import Three


def run():
    results = {}.fromkeys((-1, 0, 1), 0)
    for i in range(100):
        board = Three()
        player_1 = Player('p1', board)
        player_2 = Player('p2', board)
        players_iter = itertools.cycle([player_1, player_2])
        while True:
            player = next(players_iter)  # p1 first then p2
            action = player.choose_action()
            over, winner = board.update(action)
            board.print_board()
            if over:
                results[winner] += 1
                break
            time.sleep(0.5)
        if (i+1) % 100 == 0:
            print('result: ', results)


if __name__ == '__main__':
    run()

