# -*- coding: utf-8 -*-

import numpy as np
import torch
import random
from difflib import SequenceMatcher
from levenshtein import levenshtein

seed = 3
random.seed(seed)

# Environment simulator class
class Env:
    def __init__(self, res_dict):
        self.res_dict = res_dict
        self.turn = 0
        pass

    def reset(self):
        pass

    def feedback(self, action):
        """
        Numbers
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
        """

        # chosen position for the current number
        chosen_number = 0

        number = [self.res_dict['zero'],
                       self.res_dict['zero'] + self.res_dict['one'],
                       self.res_dict['zero'] + self.res_dict['one'] + self.res_dict['two'],
                       self.res_dict['zero'] + self.res_dict['one'] + self.res_dict['two'] + self.res_dict['three'],
                       self.res_dict['zero'] + self.res_dict['one'] + self.res_dict['two'] + self.res_dict['three'] + self.res_dict['four'],
                       self.res_dict['zero'] + self.res_dict['one'] + self.res_dict['two'] + self.res_dict['three'] + self.res_dict['four'] + self.res_dict['five'],
                       self.res_dict['zero'] + self.res_dict['one'] + self.res_dict['two'] + self.res_dict['three'] + self.res_dict['four'] + self.res_dict['five'] + self.res_dict['six'],
                       self.res_dict['zero'] + self.res_dict['one'] + self.res_dict['two'] + self.res_dict['three'] + self.res_dict['four'] + self.res_dict['five'] + self.res_dict['six'] + self.res_dict['seven'],
                       self.res_dict['zero'] + self.res_dict['one'] + self.res_dict['two'] + self.res_dict['three'] + self.res_dict['four'] + self.res_dict['five'] + self.res_dict['six'] + self.res_dict['seven'] + self.res_dict['eight'],
                       self.res_dict['zero'] + self.res_dict['one'] + self.res_dict['two'] + self.res_dict['three'] + self.res_dict['four'] + self.res_dict['five'] + self.res_dict['six'] + self.res_dict['seven'] + self.res_dict['eight'] + self.res_dict['nine']]

        probability = random.uniform(0, 1)

        if probability < 0.9:
            if action >= 0 and action < number[0]:
                chosen_number = 0
            elif action >= number[0] and action < number [1]:
                chosen_number = 1
            elif action >= number[1] and action < number [2]:
                chosen_number = 2
            elif action >= number[2] and action < number [3]:
                chosen_number = 3
            elif action >= number[3] and action < number [4]:
                chosen_number = 4
            elif action >= number[4] and action < number [5]:
                chosen_number = 5
            elif action >= number[5] and action < number [6]:
                chosen_number = 6
            elif action >= number[6] and action < number [7]:
                chosen_number = 7
            elif action >= number[7] and action < number [8]:
                chosen_number = 8
            elif action >= number[8] and action < number [9]:
                chosen_number = 9

        # for i in range(10):
        #     if self.turn == i:
        #         index = i
        #
        # if self.turn == 9:
        #     self.turn = 0
        # else:
        #     self.turn += 1

        # return chosen_number, index
        return chosen_number



# Agent class
class Agent:
    def __init__(self):
        self.num_list = np.zeros(10)
        self.my_list = []
        pass

    def reset(self):
        self.num_list = np.zeros(10)
        self.my_list = []
        pass

    def get_state(self):
        state = np.asarray(self.num_list, dtype=np.float32)
        state = torch.from_numpy(state).reshape(1, 10)
        return state

    def evaluate_reward(self, chosen_number):
        # list of target order of numbers
        target_num_list = np.arange(10)

        old_sl = -(levenshtein(target_num_list, self.num_list))

        # to check if there is anything inside agent's list
        count = len(self.my_list)

        # go to the position of next number to place, and check if
        # chosen number matches the target, then append to list
        for i in range(10):
            if count == i:
                if chosen_number == target_num_list[i]:
                    self.my_list.append(chosen_number)
                    self.num_list[i] = chosen_number

        new_sl = -(levenshtein(target_num_list, self.num_list))

        reward = new_sl - old_sl

        # giving the reward based on the numbers placed on the agent's list
        if reward == 0:
            reward -= 10

        # done and not done condition based on perfect reward
        if (self.num_list == target_num_list).all():
            done = 1
        else:
            done = 0

        return reward, done
