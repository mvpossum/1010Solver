import game, itertools, time, random, naive_alg
from random import shuffle
import itertools
import copy

def get_best_move_seqs_new(board):
    start = time.clock()
    output = []
    max_score = None
    curr_board = board.copy()
    computed_moves1 = []
    valid_moves = curr_board.get_valid_moves()
    valid_moves_p = []
    if len(curr_board.current_pieces)<3:
        return []
    for i,p in enumerate(curr_board.current_pieces):
        valid_moves_p.append(curr_board.get_valid_moves_2(p))
        shuffle(valid_moves_p[i])
    # arbitrarily large negative
    # ~ for valid_moves_p in itertools.permutations(valid_moves_p2):
    
    for move_one in valid_moves_p[0]:
        curr_board.make_move(move_one)
        for move_two in valid_moves_p[1]:
            if curr_board.is_valid_move(move_two):
                curr_board.make_move(move_two)
                for move_three in valid_moves_p[2]:
                    if curr_board.is_valid_move(move_three):
                        curr_board.make_move(move_three)
                        score = score_board(curr_board)
                        if max_score == None or score > max_score:
                            output = [[move_one,move_two,move_three]]
                            max_score = score
                        curr_board.undo_move()
                curr_board.undo_move()
        curr_board.undo_move()
        if time.clock() - start > 5:
            return output
    return output   

def get_num_free_spaces(board):
    count = 0
    for x in range(10):
        for y in range(10):
            if board.matrix[x][y] == 0:
                count += 1
    return count
    
def get_num_free_heavy(board):
    count = 0
    for x in range(10):
        for y in range(10):
            if board.matrix[x][y] > 1:
                count -= board.matrix[x][y] ** 2
    return count

def get_num_free_lines(board):
    count = 20
    for x in range(10):
        for y in range(10):
            if board.matrix[x][y] == 1:
                count -= 1
                break
    for y in range(10):
        for x in range(10):
            if board.matrix[x][y] == 1:
                count -= 1
                break
    return count

def can_place_all_pieces(board):
    curr_board = board.copy()
    biggest_pieces = [game.piece_dict['e'],game.piece_dict['i'],game.piece_dict['s']]
    score = 0
    for piece in biggest_pieces:
        curr_board.current_pieces = [piece]
        if curr_board.has_valid_moves():
            score += 150
    return score
    

def squared_continuous_spaces(board):
    score = 0
    count = 0
    for x in range(10):
        for y in range(10):
            if board.matrix[x][y] == 0:
                count += 1
            else:
                if count == 1:
                    score -= 6
                    count = 0
                else:
                    score += count**2
                    count = 0
    if count == 1:
        score -= 6
        count = 0
    else:
        score += count**2
        count = 0
    for y in range(10):
        for x in range(10):
            if board.matrix[x][y] == 0:
                count += 1
            else:
                if count == 1:
                    score -= 6
                    count = 0
                else:
                    score += count**2
                    count = 0
    if count == 1:
        score -= 6
        count = 0
    else:
        score += count**2
        count = 0
    return score

def score_board(board):
    # if can_place_all_pieces(board):
    #   return get_num_free_lines(board)*100+get_num_free_spaces(board)+200
    # return get_num_free_lines(board)*100+get_num_free_spaces(board)
    return squared_continuous_spaces(board)+can_place_all_pieces(board)*2+get_num_free_heavy(board)


move_queue = []
def refresh_move_queue(board):
    global move_queue
    best_moves = get_best_move_seqs_new(board)
    if best_moves != []:
        move_queue = best_moves[0]

def get_move(board):
    global move_queue
    if move_queue == []:
        refresh_move_queue(board)
    if move_queue == []:
        move_queue = [naive_alg.get_move(board)]
    move = move_queue.pop(0)
    return move
