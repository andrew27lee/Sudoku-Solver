from src.solving_objects.Sudoku import *


def solve_grid(sudo):
    while not sudo.is_filled():
        if sudo.should_make_hypothesis():
            x, y, possible_values_hyp = sudo.best_hypothesis()
            if not possible_values_hyp:
                return False, None
            for val in possible_values_hyp:
                new_sudo = Sudoku(sudo=sudo)
                new_sudo.apply_hypothesis_value(x, y, val)
                ret, solved_sudo = solve_grid(new_sudo)
                if ret:
                    return True, solved_sudo
                else:
                    del new_sudo
            return False, None
        else:
            ret = sudo.apply_unique_possibilities()
            if ret is False:
                del sudo
                return False, None
    return True, sudo


def main_solve_grids(grids):
    finished_grids = []
    for grid in grids:
        finished_grids.append(main_solve_grid(grid))

    if all(elem is None for elem in finished_grids):
        return None
    return finished_grids


def main_solve_grid(grid):
    if grid is None:
        return None
    sudo = Sudoku(grid=grid)
    ret, finished_sudo = solve_grid(sudo)
    if ret:
        return finished_sudo.grid
    else:
        return None
