from c4.state import C4State

class BudgetExceededError(Exception):
    """
    Custom exception for when the computational budget is exhausted.
    """
    pass

def get_column_weights(cols):
    center = cols // 2
    weights = [0] * cols
    for col in range(cols):
        # weight is inversely proportional to the distance from the center
        distance_from_center = abs(center - col)
        weights[col] = max(0, 200 - 40 * distance_from_center)
    return weights

def evaluation_function(state: C4State, max_player: int):
    """
    Heuristic inspired from the work of Kang et. al. (2019)

    Priority:
    
    1. feat_1 (inf): Check if we have won the game.
    2. feat_2' (-inf): Check if leads to opponent winning next turn.
    3. feat_2 (inf): Check if leads to us winning next turn.
    4. (feat2 + feat_3 + feat_4) - (feat_2' + feat_3' + feat_4'): Compute HV.
    """

    min_player = 3 - max_player

    # feature 1
    if state.winner == max_player:
        return float('inf')
    elif state.winner == min_player:
        return float('-inf')
    
    # if it leads to an automatic loosing position
    against = feature_2(state, min_player)
    if against == float('inf'):
        return -against
    
    # if it leads to an automatic winning position
    in_favour = feature_2(state, max_player)
    if in_favour == float('inf'):
        return in_favour

    in_favour += feature_3(state, max_player) + feature_4(state, max_player)
    against += feature_3(state, min_player) + feature_4(state, min_player)
    
    return in_favour - against

def count_spaces_available(x, y, dx, dy, state: C4State):
    """
    Count free spaces immediately available in a given direction.
    """
    count = 0
    while state.available_immediately(x, y):
        count += 1
        x += dx
        y += dy
    return count

def feature_2(state: C4State, player):
    coord_dict = state.find_sequence(state.connect-1, player)
    util = 0
    for direction, coords in coord_dict.items():
        dx, dy = state.directions[direction]
        for ((x_s, y_s), (x_e, y_e)) in coords:
            count = 0
            if state.available_immediately(x_s-dx, y_s-dy):      # check
                count += 1                                              # both
            if state.available_immediately(x_e+dx, y_e+dy):      # sides
                count += 1                                              # of sequence
            
            if count == 2:
                return float('inf')
            elif count == 1:
                util = 900000 
    return util

def feature_3(state: C4State, player):
    coord_dict = state.find_sequence(state.connect-2, player)
    util = 0
    for direction, coords in coord_dict.items():
        dx, dy = state.directions[direction]
        for ((x_s, y_s), (x_e, y_e)) in coords:
            
            start_spaces = count_spaces_available(x_s, y_s, -dx, -dy, state)
            end_spaces = count_spaces_available(x_e, y_e, +dx, +dy, state)

            if start_spaces != 0 and end_spaces != 0:
                return 50000
            util = (max(max(start_spaces, end_spaces)-1, 0)) * 10000 
            
    return util

def feature_4(state: C4State, player):
    column_weights = get_column_weights(state.cols)
    coord_dict = state.find_sequence(1, player)
    util = 0
    for _, coords in coord_dict.items():
        for coord in coords:
            util += column_weights[coord[0][1]]
    return util