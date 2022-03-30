import numpy as np


class Sudoku:
    def __init__(self, sudo=None, grid=None):
        self.possible_values_grid = np.empty((9, 9), dtype=list)
        if sudo is None:
            self.grid = np.zeros((9, 9), dtype=int)
            self.count_possible_grid = np.zeros((9, 9), dtype=int)
            self.init_sudo(grid)
        else:
            self.grid = sudo.grid.copy()
            for y in range(9):
                for x in range(9):
                    self.possible_values_grid[y, x] = sudo.possible_values_grid[y, x].copy()
            self.count_possible_grid = sudo.count_possible_grid.copy()
    
    def apply_hypothesis_value(self, x, y, value):
        self.grid[y, x] = value
        self.possible_values_grid[y, x] = []
        self.count_possible_grid[y, x] = 0

        for y2 in range(9):
            for x2 in range(9):
                if is_affected(x, y, x2, y2) and self.grid[y2, x2] == 0:
                    list_possible_values = self.possible_values_grid[y2, x2]
                    if value in list_possible_values:
                        list_possible_values.remove(value)
                        new_len = len(list_possible_values)
                        self.count_possible_grid[y2, x2] = new_len
    
    def init_sudo(self, grid):
        for y in range(9):
            for x in range(9):
                value = grid[y][x]
                self.grid[y, x] = value
                if value == 0:
                    self.possible_values_grid[y, x] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                    self.count_possible_grid[y, x] = 9
                else:
                    self.possible_values_grid[y, x] = []
        self.get_possible_values()
    
    def is_filled(self):
        return 0 not in self.grid
    
    def get_possible_values(self):
        for y in range(9):
            for x in range(9):
                if self.grid[y, x] != 0:
                    continue
                possible_values = self.get_1_possible_values(x, y)
                self.possible_values_grid[y, x] = possible_values
                self.count_possible_grid[y, x] = len(possible_values)
    
    def get_1_possible_values(self, x, y):
        possible_values = self.possible_values_grid[y, x]
        self.check_line(y, possible_values)
        self.check_column(x, possible_values)
        self.check_square(x, y, possible_values)
        return possible_values
    
    def check_line(self, y, possible_values):
        line = self.grid[y, :]
        for value in reversed(possible_values):
            if value in line:
                possible_values.remove(value)
    
    def check_column(self, x, possible_values):
        column = self.grid[:, x]
        for value in reversed(possible_values):
            if value in column:
                possible_values.remove(value)
    
    def check_square(self, x, y, possible_values):
        x1 = 3 * (x // 3)
        y1 = 3 * (y // 3)
        x2, y2 = x1 + 3, y1 + 3
        square = self.grid[y1:y2, x1:x2]
        for value in reversed(possible_values):
            if value in square:
                possible_values.remove(value)
    
    def apply_and_actualize(self, x, y, value):
        self.grid[y, x] = value
        self.possible_values_grid[y, x] = []
        self.count_possible_grid[y, x] = 0

        for y2 in range(9):
            for x2 in range(9):
                if is_affected(x, y, x2, y2) and self.grid[y2, x2] == 0:
                    list_possible_values = self.possible_values_grid[y2, x2]
                    if value in list_possible_values:
                        list_possible_values.remove(value)
                        new_len = len(list_possible_values)
                        if new_len == 0:
                            return False
                        self.count_possible_grid[y2, x2] = new_len
        return True
    
    def apply_unique_possibilities(self):
        for y in range(9):
            for x in range(9):
                if self.grid[y, x] == 0 and self.count_possible_grid[y, x] == 1:
                    value = self.possible_values_grid[y, x][0]
                    if not self.apply_and_actualize(x, y, value):
                        return False

        return True
    
    def should_make_hypothesis(self):
        return 1 not in self.count_possible_grid
    
    def best_hypothesis(self):
        count_less_options = 9
        best_x = 0
        best_y = 0
        for y in range(9):
            for x in range(9):
                if self.grid[y, x] != 0:
                    continue
                if self.count_possible_grid[y, x] == 2:
                    return x, y, self.possible_values_grid[y, x]
                elif self.count_possible_grid[y, x] < count_less_options:
                    best_x, best_y = x, y
                    count_less_options = self.count_possible_grid[y, x]
                    if count_less_options == 0:
                        return None, None, []

        return best_x, best_y, self.possible_values_grid[best_y, best_x]
    
    def verify_result(self):
        for y in range(9):
            for x in range(9):
                grid = self.grid.copy()
                grid[y, x] = 0
                line = grid[y, :]
                column = grid[:, x]
                x1 = 3 * (x // 3)
                y1 = 3 * (y // 3)
                x2, y2 = x1 + 3, y1 + 3
                square = grid[y1:y2, x1:x2]
                val = self.grid[y, x]
                if val in line or val in column or val in square:
                    return False
        return True
    

def verify_viable_grid(grid_tested):
    for y in range(9):
        for x in range(9):
            if grid_tested[y, x] == 0:
                continue
            grid = grid_tested.copy()
            grid[y, x] = 0
            line = grid[y, :]
            column = grid[:, x]
            x1 = 3 * (x // 3)
            y1 = 3 * (y // 3)
            x2, y2 = x1 + 3, y1 + 3
            square = grid[y1:y2, x1:x2]
            val = grid_tested[y, x]
            if val in line or val in column or val in square:
                return False

    return True


def is_affected(x1, y1, x2, y2):
    if x1 == x2:
        return True
    if y1 == y2:
        return True

    if x1 // 3 == x2 // 3 and y1 // 3 == y2 // 3:
        return True

    return False
