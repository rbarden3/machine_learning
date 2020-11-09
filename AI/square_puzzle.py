# Course: CS4242
# Student name: Raleigh Barden
# Student ID: 000-86-1747
# Assignment #: #1
# Due Date: 09/25/2020  

# Description: Create a program to solve an 8 Square Puzzle using an A* algorythm. It should accept any input, goal pairs.

# Developer Environment: VS Code

#%%
import copy
import random
import numpy as np

#Prints an Arrow
def print_arrow():
    for _ in range(4):
        print('      V')
    print()

#This is a function that turns a square 2d array into an unicode grid. I made it for a chess game and thought it would be suitable here
def print_grid(input_array):
    chars = {   
            "line":     {"horizontal":"\u2500", "vertical":"\u2502"},
            "corner":   {"up_right":"\u2514","up_left":"\u2518", "down_right":"\u250C", "down_left":"\u2510", "cross":"\u253C"},
            "t":        {"right":"\u251C", "left":"\u2524", "down":"\u252C", "up":"\u2534"}    
        }

    in_data = copy.deepcopy(input_array)
    grid = ""
    num_rows = len(in_data)
    num_cols = len(in_data[0])

    for row_ind, row in enumerate(in_data):
        for cell_ind, cell in enumerate(row):
            if cell is None:
                in_data[row_ind][cell_ind] = ' '
            else:
                in_data[row_ind][cell_ind] = str(cell)

    data_wid = len(in_data[0][0]) + 2

    hz_line = chars["line"]["horizontal"] *data_wid
    new_line = "\n"
    top_row =  chars["corner"]["down_right"] + hz_line + (chars["t"]["down"] + hz_line) *(num_cols-1) + chars["corner"]["down_left"] + new_line
    mid_row = chars["t"]["right"]+ hz_line + (chars["corner"]["cross"] + hz_line) *(num_cols-1) + chars["t"]["left"] + new_line
    bot_row = chars["corner"]["up_right"] + hz_line + (chars["t"]["up"] + hz_line) *(num_cols-1) + chars["corner"]["up_left"] + new_line
    for ind, val in enumerate(in_data):
        if (ind == 0):
            grid += top_row 
        else:
            grid += mid_row

        dta_row = ""
        for _, cell in enumerate(val): 
            dta_row += chars["line"]["vertical"] + ' ' + str(cell) + ' '
        dta_row += chars["line"]["vertical"] + new_line
        grid += dta_row
    grid += bot_row

    #print(grid)
    print (grid)

#Example Fuction Showing Manual Sorting
def run_example_sort():
    sorted_square = [[1,2,3],[8,None,4],[7,6,5]]
    unsorted_square = [[2,8,3],[1,6,4],[7,None,5]]
    print_grid(sorted_square)
    # print("Unsorted Array:")
    # print(print_grid(unsorted_square))

    print("example sorting process:\n")

    print_grid(unsorted_square)
    print_arrow()

    print_grid([ [2,8,3],[1,None,4],[7,6,5]])
    print_arrow()

    print_grid([[2,None,3],[1,8,4],[7,6,5]])
    print_arrow()

    print_grid([[None,2,3],[1,8,4],[7,6,5]])
    print_arrow()

    print_grid([[1,2,3],[None,8,4],[7,6,5]])
    print_arrow()
    print_grid([[1,2,3],[8,None,4],[7,6,5]])

# Creates a 2d array with the values 1-8 filled in randomly
def create_square_puzzle():
    #initialize data
    values = [1,2,3,4,5,6,7,8]
    puzzle = [[None,None,None],[None,None,None],[None,None,None]]
    available = {0:[0,1,2], 1:[0,1,2], 2:[0,1,2]}

    # While there are still values in data
    while (len(values) > 0):
        # Remove a random value from data
        val = values.pop( random.randrange( len(values) ) )

        # Decide a random row to place it in
        row = random.choice(list(available.keys()))
        # Decide a random col to place it in
        col = available[row].pop( random.randrange( len(available[row]) ) )
        # if there are no more open spaces in a row, delete it
        if(len(available[row]) == 0):
            del available[row]
        #place the value in its assigned spot
        puzzle[row][col] = val
    return puzzle


# Find the row and column values for the blank space in the square
def get_open_ind(in_puzzle):
    # Copy square to avoid reference issues
    grid_arr = copy.deepcopy(in_puzzle)
    # initialize row and col
    row = col = -1
    #loop through rows
    for row_ind, row_val in enumerate(grid_arr):
        if (col == -1):
            try:
                # try to find None in the row, if it is not there, continue to next row
                # if it is there, set row and col
                col = row_val.index(None)
                row = row_ind
                #print("open_ind:",row, col)
            except ValueError:
                pass
        else:
            break
    return (row, col)

# returns possible directions for the blank space to move
def get_possible_moves(puzzle):
    row, col = get_open_ind(puzzle)
    possible_moves = {'up','down','left','right'}
    if(row == 0):
        possible_moves.remove('up')
    if(row == 2):
        possible_moves.remove('down')
    if(col == 0):
        possible_moves.remove('left')
    if(col == 2):
        possible_moves.remove('right')

    return possible_moves

# Moves the blank space up
def up(in_puzzle):
    row, col = get_open_ind(in_puzzle)
    puzzle = copy.deepcopy(in_puzzle)
    temp = puzzle[row-1][col]
    puzzle[row-1][col] = puzzle[row][col]
    puzzle[row][col] = temp
    return puzzle

# Moves the blank space down
def down(in_puzzle):
    row, col = get_open_ind(in_puzzle)
    puzzle = copy.deepcopy(in_puzzle)
    temp = puzzle[row+1][col]
    puzzle[row+1][col] = puzzle[row][col]
    puzzle[row][col] = temp
    return puzzle

# Moves the blank space left
def left(in_puzzle):
    row, col = get_open_ind(in_puzzle)
    puzzle = copy.deepcopy(in_puzzle)
    temp = puzzle[row][col-1]
    puzzle[row][col-1] = puzzle[row][col]
    puzzle[row][col] = temp
    return puzzle

# Moves the blank space right
def right(in_puzzle):
    row, col = get_open_ind(in_puzzle)
    puzzle = copy.deepcopy(in_puzzle)
    temp = puzzle[row][col+1]
    puzzle[row][col+1] = puzzle[row][col]
    puzzle[row][col] = temp
    return puzzle

# Returns the inversion count of given square
def get_inversion_count(square):
    square = np.array(square).flatten()
    count = 0
    for r in range(len(square)):
        for c in range(r+1, len(square)):
            if (square[r] not in [None,0] and square[c] not in [None,0] and square[r] > square[c]):
                count +=1
    #print(count)                
    return count

# returns whether or not the puzzle is solvable
def solvable(unsorted_sq, goal_sq):
    unsorted_inv = get_inversion_count(unsorted_sq)
    goal_inv = get_inversion_count(goal_sq)
    # The Inversion count of both the unsorted array and goal array need to be either even or odd to be solvable
    if unsorted_inv % 2 == goal_inv % 2:
        return True
    else:
        return False
        

# %%
#returns the summation of the distance between all cells in square and goal square
def h_score(unsorted_sq, goal_sq):
    #Flatten Squares
    flat_unsorted = np.array(unsorted_sq).flatten()
    flat_goal = np.array(goal_sq).flatten()

    values = [1,2,3,4,5,6,7,8, None]
    manhattan_distances = []

    for value in values:
        #Find Unsorted Value Point
        row = list(flat_unsorted).index(value) / 3
        col = list(flat_unsorted).index(value) % 3
        unsorted_point = (col,row)
        # Find Goal Value Point
        row = list(flat_goal).index(value) / 3
        col = list(flat_goal).index(value) % 3
        goal_point = (col,row)
        #Calculate Manhattan Distance
        manhattan_distances.append( abs(unsorted_point[0]-goal_point[0]) + abs(unsorted_point[1]-goal_point[1]) )

    # Calculate Summation of Manhattan Distances and Return
    manhattan_summation = 0
    for distance in manhattan_distances:
        manhattan_summation += distance
    return manhattan_summation

#node for square tree
class Node:
    def __init__(self, square, goal, g_score, f_score, move, parent= None):
        self.move = move
        self.square = square
        self.goal = goal
        self.g_score = g_score
        self.f_score = f_score
        self.parent = parent
        

    # generates children based on possible moves
    def gen_children(self, parent):
        possible_moves = get_possible_moves(self.square)
        children = []
        for move in possible_moves:

            test_sq = None
            if (move == 'up'):
                test_sq = up(self.square)
            elif(move == 'down'):
                test_sq = down(self.square)
            elif(move == 'left'):
                test_sq = left(self.square)
            elif(move == 'right'):
                test_sq = right(self.square)
            goal = self.goal
            g_score = self.g_score + 1
            #h_score(test_sq, goal)
            f_score =  h_score(test_sq, goal) + g_score

            children.append(Node(test_sq, goal, g_score, f_score, move, parent ) )
        return children

#returns a nodes f score, used for sorting
def get_f (Nod:Node):
    return Nod.f_score

#A* Algorythm
def A_Star(initial, goal):
    # To Run, Puzzle Must Be Solvable
    assert solvable(initial, goal)
    open = []
    closed = []
    move_count = 0
    head = Node(initial, goal, 0, h_score(initial, goal), move='Initial')
    open.append(head)
    #limit = 1000

    sequence = []


    while (True):
        # Sort so that the next node will be have the best f_score
        open.sort(key=get_f)
        #Set Current node to the first value (lowest f-score)
        current_node = open[0]

        # Print Data about Move
        print("Move", move_count, ":")
        print("g-score:", current_node.g_score, "h-score:", current_node.f_score-current_node.g_score, "f-score:", current_node.f_score )
        print_grid(current_node.square)
        # if h-score is 0, we have reached goal, so stop looking
        if (h_score(current_node.square, goal) == 0):
            while (True):
                #build sequence by working backwards through the tree
                sequence.insert(0, current_node.move)
                if (current_node.parent != None):
                    current_node = current_node.parent
                else: break
            break
        move_count += 1
        # append children to list
        for child in current_node.gen_children(current_node):
            open.append(child)
        # append the node just used to closed
        closed.append(open[0])
        # delete the last used value
        del open[0]
    return sequence
        

# %%
# Using Example Puzzle:
goal_square = [[1,2,3],[8,None,4],[7,6,5]]
initial_square = [[2,8,3],[1,6,4],[7,None,5]]
sequence = A_Star(initial_square, goal_square)
print("Initial Square:")
print_grid(initial_square)
print("Goal Square:")
print_grid(goal_square)
print("Sequence:")
seq_str = ""
for move in sequence:
    seq_str += 'Blank Tile Moves ' + move + ', '
print(seq_str)
#%%
#Using Random Puzzle:
goal_square = create_square_puzzle()
initial_square = create_square_puzzle()
while (not solvable(initial_square, goal_square)):
    goal_square = create_square_puzzle()
    initial_square = create_square_puzzle()

sequence = A_Star(initial_square, goal_square)
print("Initial Square:")
print_grid(initial_square)
print("Goal Square:")
print_grid(goal_square)
print("Sequence:")
seq_str = ""
for move in sequence:
    seq_str += 'Blank Tile Moves ' + move + ', '
print(seq_str)

#%%