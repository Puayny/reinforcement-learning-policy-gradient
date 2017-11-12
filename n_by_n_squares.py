# a game with n*n squares.
# output is 0 is no squares lit, 1 if left square lit, 2 if right square lit. Only 1 square will be lit at a time.
# game ends after 10 rounds.

import random
import numpy as np

class NByNSquares:
    def __init__(self, grid_size=2, max_rnd=10, verbose=False):
        self.grid_size = grid_size
        self.max_rnd = max_rnd
        self.curr_rnd = 0
        self.curr_ans = None
        self.curr_grid = None
        self.score_history = [0 for i in range(max_rnd)]
        self.verbose = verbose

        self.random_grid()

    def __str__(self):
        str_repr = ''
        for row in self.curr_grid:
            for cell in row:
                str_repr += ' {} '.format(cell)
            str_repr += '\n'
        str_repr += 'Curr score: {}\n'.format(sum(self.score_history))
        return str_repr

    def random_grid(self):
        self.curr_grid = [[0 for i in range(self.grid_size)] for j in range(self.grid_size)]
        rand_row = random.randint(0,self.grid_size-1)
        rand_col = random.randint(0,self.grid_size-1)
        self.curr_grid[rand_row][rand_col] = 1
        self.curr_ans = rand_col

    def get_game_grid(self):
        return self.curr_grid

    def get_vector_repr(self):
        return np.array(self.get_game_grid()).flatten()

    def get_score_history(self):
        return self.score_history

    def get_curr_round(self):
        return self.curr_rnd

    def check_action(self, action):
        if action == self.curr_ans:
            self.score_history[self.curr_rnd] = 1
        else:
            self.score_history[self.curr_rnd] = -1
        if self.verbose:
            print(self)
            print('Answer ({}) is {}\n'.format(action, action == self.curr_ans))
        return self.score_history[self.curr_rnd]

    def take_action(self, action):
        reward = self.check_action(action)
        is_game_over = self.check_game_over()
        if is_game_over and self.verbose: print('Gameover\n')
        self.next_round()
        return reward, is_game_over


    def check_game_over(self):
        return self.curr_rnd + 1 >= self.max_rnd

    def next_round(self):
        self.curr_rnd += 1
        self.random_grid()




if __name__ == '__main__':
    game = NByNSquares(verbose=True)
    print(game)
    game.take_action(0)
    print(game)
    game.take_action(1)
    print(game)
    game.take_action(0)