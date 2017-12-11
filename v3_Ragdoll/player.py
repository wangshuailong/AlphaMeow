#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import random
from mcts import MCTS
from rave import MCRave


class Player(object):
    def __init__(self, name, board, method=None):
        self.name = name
        self.board = board
        self.method = method

    def choose_action(self):
        if self.method == 'mcts':
            agent = MCTS(name='mcts', board=self.board)
            action = agent.choose_action()
        elif self.method == 'mcrave':
            agent = MCRave(name='mcrave', board=self.board)
            action = agent.choose_action()
        else:
            available_actions = self.board.legal_positions
            action = random.choice(available_actions)
        return action

