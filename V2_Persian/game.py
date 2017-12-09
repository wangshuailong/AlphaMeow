#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import itertools
import random
from itertools import groupby
from operator import itemgetter


class Three(object):

    num_players = 2
    target = 3
    empty_note = -1

    def __init__(self, width=6):

        self.width = width

        self.board, self.steps, self.player = None, None, None
        self.legal_positions = None
        self.position_taken = None
        self.reset()

    def reset(self):
        states = [[(x, y) for x in range(0, self.width)] for y in range(0, self.width)]
        self.board = dict().fromkeys(sorted(list(itertools.chain(*states))), self.empty_note)  # -1 empty, 1 or 0 player

        legal_tuple_positions = [positions for positions, player in self.board.items() if player == self.empty_note]
        self.legal_positions = list(map(self.tuple2number, legal_tuple_positions))

        self.steps = 0  # random.choice((range(self.num_players)))
        self.player = self.current_player()

        self.position_taken = []

    def current_player(self):
        return self.steps % self.num_players

    def next_player(self):
        return (self.steps+1) % self.num_players

    def update(self, action):
        # done = False
        if action not in self.legal_positions:
            raise ValueError('Please Choose A Legal Action.')

        self.update_board(action)  # update board
        self.update_legal_actions(action)  # update legal actions pool
        self.steps += 1  # update turn for player

        over, winner = self.game_over()
        return over, winner

    def update_board(self, action):
        player = self.current_player()
        tuple_action = self.number2tuple(action)
        self.board[tuple_action] = player

    def update_legal_actions(self, action):
        self.legal_positions.remove(action)

    def game_over(self):
        """
        tell if there is winner
        :return: if someone wins and winner, -1 continue 1 or 0 player win
        """
        winner = -1
        for p in range(self.num_players):
            pieces = [positions for positions, player in self.board.items() if player == p]
            # row and column
            for i in range(0, self.width):
                row_pieces = [piece[1] for piece in pieces if piece[0] == i]
                over = self.consecutive_number(row_pieces)
                if over:
                    # print('Game Over. The Winner Is: ', p)
                    return True, p
                col_pieces = [piece[0] for piece in pieces if piece[1] == i]
                over = self.consecutive_number(col_pieces)
                if over:
                    # print('Game Over. The Winner Is: ', p)
                    return True, p
            # TODO: diagonal
        if len(self.legal_positions) == 0:
            print('Game Over. It Is A Draw.')
            print('Draw winner: ', winner)
            return True, winner
        else:
            return False, winner

    def consecutive_number(self, data):
        for _, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
            consecutive_pieces = list(map(itemgetter(1), g))
            if len(consecutive_pieces) >= self.target:
                return True

    def tuple2number(self, tuple_state):
        number_state = (tuple_state[0])*self.width + tuple_state[1]
        return number_state

    def number2tuple(self, number_state):
        tuple_state = (number_state//self.width, number_state % self.width)
        return tuple_state

    def print_board(self):
        print('\n\n')
        for i in range(self.width):
            for j in range(self.width):
                loc = (i, j)
                if self.board[loc] == 1:
                    print('X'.center(7), end='')
                elif self.board[loc] == 0:
                    print('O'.center(7), end='')
                else:
                    print('_'.center(7), end='')
            print('\n\r\n\r')

