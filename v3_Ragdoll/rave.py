#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import copy
import random
import datetime
import math


class MCRave(object):
    def __init__(self, name, board, simulation_time=0.5, max_moves=100, c=1.96, k=100):
        self.name = name
        self.board = board

        self.simulation_time = datetime.timedelta(seconds=simulation_time)
        self.max_moves = max_moves
        self.C = c
        self.K = k

        self.max_depth = 1

        self.move_records = {}
        self.win_records = {}
        self.move_records_rave = {}
        self.win_records_rave = {}

    def choose_action(self):
        available_actions = self.board.legal_positions
        player = self.board.current_player()

        if len(available_actions) == 1:
            action = available_actions[0]
            return action

        simulation_times = 0
        begin_time = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin_time <= self.simulation_time:
            board4simulation = copy.deepcopy(self.board)
            player4simulation = copy.deepcopy(player)
            self.run_simulation(board4simulation, player4simulation)
            simulation_times += 1
        # print('simulation times: ', simulation_times)

        win_prob, action = max((self.win_records.get((player, move), 0) /
                                self.move_records.get((player, move), 1), move) for move in available_actions)

        # self.prune()

        return action

    def run_simulation(self, board, player):
        simulation_visits = set()
        expand = True
        available_actions = board.legal_positions

        for t in range(self.max_moves):
            if all(self.move_records.get((player, action)) for action in available_actions):
                # log_total = math.log(
                #     sum(self.move_records[(player, move)] for move in available_actions))
                # value, action = max(
                #     ((self.win_records[(player, move)] / self.move_records[(player, move)]) +
                #      math.sqrt(self.C * log_total / self.move_records[(player, move)]), move)
                #     for move in available_actions)   # UCB
                #  UCT RAVE: (1-beta)*MC + beta*AMAF + UCB
                value, action = max(
                    ((1-math.sqrt(self.K/(3 * self.move_records_rave[move] + self.K))) * (self.win_records[(player, move)] / self.move_records[(player, move)]) +
                     math.sqrt(self.K/(3 * self.move_records_rave[move] + self.K)) * (self.win_records_rave[move][player] / self.move_records_rave[move]) +
                     math.sqrt(self.C * math.log(self.move_records_rave[move]) / self.move_records[(player, move)]), move)
                    for move in available_actions)
            else:
                action = random.choice(available_actions)
            win, winner = board.update(action)

            if expand and (player, action) not in self.move_records:
                expand = False
                self.move_records[(player, action)] = 0
                self.win_records[(player, action)] = 0

                if action not in self.move_records_rave:
                    self.move_records_rave[action] = 0
                if action in self.win_records_rave:
                    self.win_records_rave[action][player] = 0
                else:
                    self.win_records_rave[action] = {player: 0}

            simulation_visits.add((player, action))  # add to simulation record for back-propagation

            if win or len(available_actions) == 0:
                break

            player = board.current_player()  # change player for next turn

        for player, action in simulation_visits:
            if (player, action) in self.move_records:
                self.move_records[(player, action)] += 1
                if player == winner:
                    self.win_records[(player, action)] += 1
            if action in self.move_records_rave:
                self.move_records_rave[action] += 1
                if winner in self.win_records_rave[action]:
                    self.win_records_rave[action][winner] += 1



