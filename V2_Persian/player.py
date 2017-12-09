#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import random


class Player(object):
    def __init__(self, name, board):
        self.name = name
        self.board = board

    def choose_action(self):
        available_actions = self.board.legal_positions
        action = random.choice(available_actions)
        return action

