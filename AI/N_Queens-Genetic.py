# Course: CS4242
# Student name: Raleigh Barden
# Student ID: 000-86-1747
# Assignment #: #2
# Due Date: 11/08/2020  


from collections import Counter
from random import randint, randrange, choice
import copy

# 1 Queen per column assumed
board = [] # Represents N x N board of Queens. Index + 1 represents column No. Value represents Row No.

#This is a function that turns a square 2d array into an unicode grid. I made it for a chess game and thought it would be suitable here for visualization
def print_grid(input_array):
    chars = {   # Characters for building a grid
            "line":     {"horizontal":"\u2500", "vertical":"\u2502"},
            "corner":   {"up_right":"\u2514","up_left":"\u2518", "down_right":"\u250C", "down_left":"\u2510", "cross":"\u253C"},
            "t":        {"right":"\u251C", "left":"\u2524", "down":"\u252C", "up":"\u2534"}    
        }

    in_data = copy.deepcopy(input_array) # creates a copy of the input array to avoid relation issues
    grid = "" # initialize grid as empty string
    num_cols = len(in_data[0]) # find number of columns in grid

    for row_ind, row in enumerate(in_data): # loop through every cell in the grid
        for cell_ind, cell in enumerate(row):
            if cell is None: # if the cell is empty, replace it with a space
                in_data[row_ind][cell_ind] = ' '
            else: # Otherwise convert the value in the cell to a string
                in_data[row_ind][cell_ind] = str(cell)

    data_wid = len(in_data[0][0]) + 2 # finds the width of the objects in the Grid. Ex) Q is 1 width. adds 2 for spacing

    hz_line = chars["line"]["horizontal"] *data_wid # set up the horizontal line pieces for the grid
    new_line = "\n" # just done for readibility

    # Three row variables needed. top, middle, and bottom. 
    top_row =  chars["corner"]["down_right"] + hz_line + (chars["t"]["down"] + hz_line) *(num_cols-1) + chars["corner"]["down_left"] + new_line # create top row var
    mid_row = chars["t"]["right"]+ hz_line + (chars["corner"]["cross"] + hz_line) *(num_cols-1) + chars["t"]["left"] + new_line # create middle row variable
    bot_row = chars["corner"]["up_right"] + hz_line + (chars["t"]["up"] + hz_line) *(num_cols-1) + chars["corner"]["up_left"] + new_line # create bottom row var
    for ind, val in enumerate(in_data):
        if (ind == 0):# If the row is the first in the grid, add the top row
            grid += top_row 
        else: # Otherwise add a middle row 
            grid += mid_row

        dta_row = "" # data row has the actual values in it, and vertical seperator lines
        for _, cell in enumerate(val): # add the values of all cells along with spacing and seperators
            dta_row += chars["line"]["vertical"] + ' ' + str(cell) + ' '
        dta_row += chars["line"]["vertical"] + new_line # Add the final vertical sepertor in the line
        grid += dta_row # add the data row to the grid
    grid += bot_row # after the loop, add the bottom row to finish visualiztion of the graph

    # grid is a string so that the unicode values fit together
    print (grid)

def convert_board_to2D(board): # used to convert list of queen positions to a 2d list with spaces and "Q"s
    board_size = len(board)
    sideways_board_2d = [] # Grid will initially be sizeways
    board_2d = []

    for _, val in enumerate(board): # creates transposed 2d_grid
        col = []
        for row in range(board_size):
            if (row+1 == val):
                col.append('Q')
            else:
                col.append(' ')
        sideways_board_2d.append(col)
    
    for r in range(board_size): #transpose grid
        row = []
        for c in range(board_size):
            row.append(sideways_board_2d[c][r])
        board_2d.append(row)
    
    return board_2d

def gen_board(N = 8): # Generates a board of size N and fills it with random values between 1 and N
    board = []
    for _ in range(N):
        board.append(randint(1,N))
    return board
    
def get_duplicates(board): # Returns elements in a list that have duplicates
    return [val for val, count in Counter(board).items() if count > 1]

def board_score(board:list): # Returns the score of a board. 
    score = 0 #Initialize score as 0
    direct_diag_vals = [] #List used to store the row value - col value, which allows us to check if more than one queen is diagonal from eachother in the diaganol where r and c increase together.
    indirect_diag_vals = [] # list used to store the row - col + (2*col) value, which is used to check the diagonal where row decreases as col increases
    for ind, val in enumerate(board):# Fill diag_vals with row, col difference
        col = ind+1
        row = val
        direct_diag_vals.append(row-col)
        indirect_diag_vals.append(row-col+(2*col))

    row_dupes = get_duplicates(board) # save row duplicate values
    direct_diag_dupes = get_duplicates(direct_diag_vals) # save duplicate values for the diagonal where row and col increase together
    inverse_diag_dupes = get_duplicates(indirect_diag_vals) # save duplicate values for the diagonal where row decreases while col increases

    for ind, val in enumerate(board): # loop through the board and assign score
        if val not in row_dupes and direct_diag_vals[ind] not in direct_diag_dupes and indirect_diag_vals[ind] not in inverse_diag_dupes: 
            score +=1 # if the Queen is not in the same row or diagonal as another queen, add a point to score

    return score # Returns Number of queens that are in a valid position (not being attacked).

def cross_over(parents, n_offspring): # "Reproduction" method to create new children boards
    n_parents = len(parents)
    offspring = []# Declare offspring list
    for _ in range(n_offspring): # Do For Each Child

        p1_ind = randrange(n_parents) # Index of 1st Parent
        p2_ind = randrange(n_parents) # Index of 2nd Parent

        while(p2_ind != p1_ind): # Make sure the parents are different
            p2_ind = randrange(n_parents)

        # Set the Parents using their index value
        p1 = parents[p1_ind] 
        p2 = parents[p2_ind]

        child = [] # Declare child list
        for ind, p1_val in enumerate(p1): # Iterate through parent 1
            p2_val = p2[ind] # Set Corresponding Parent 2 Value

            # Randomly decide if the child gets parent 1 or parent 2's trait
            if (choice([True, False])): 
                child.append(p1_val)
            else:
                child.append(p2_val)
        # Append the child to the offspring list
        offspring.append(child)
    return offspring

def mutate_parent(parent, n_mutations):
    board_size = len(parent) # Board is a square N x N

    for _ in range(n_mutations):
        parent[randrange(board_size)] = randint(1, board_size) # Set a random index to a random value

    return parent

def mutate_gen(parent_gen, n_mutations):
    mutated_parents = []
    for parent in parent_gen:
        mutated_parents.append(mutate_parent(parent,n_mutations)) # perform mutation

    return mutated_parents

def is_solved(board): # Checks if the board is solved
    if (board_score(board) == len(board)): # For a board to be solved, the score is equal to the board size
        return True
    else:
        return False

def get_second_ind(elem): # Key for sorting, allows sorting based of of index 1 in a 2D object
    return elem[1]

def get_best(parent_gen, n_best, with_score = False): # Returns the list of best boards, optionally also returns their score
    score = [] # Declare score list
    for parent in parent_gen: # Find the score for each patient
        score.append( (parent, board_score(parent)) ) # append the board along with its score to the score list

    score.sort(reverse=True, key=get_second_ind) # sort the score list by the score in descending order
    new_gen = [] # Declare list for new generation
 
    if not with_score: # if the return does not need score, only append the board to new_gen
        for ind in range(n_best):
            new_gen.append(score[ind][0])  
    else:
        for ind in range(n_best): # if the return requires score, append the board and score to new_gen
            new_gen.append(score[ind])  

    return new_gen

def get_next_gen(parent_gen, gen_size, n_best, n_mutations):
    solved = [] # declare list for solved boards
    new_gen =[] # declare list for the new generation of boards

    best_boards = get_best(parent_gen, n_best) # get the n_best best boards

    for board in best_boards:
        if is_solved(board):# Seperate solved boards so they arent mutated
            solved.append(board) # allows for solved boards not to be mutated
        else:
            new_gen.append(board)

    new_gen.extend( cross_over(best_boards, n_offspring=gen_size-n_best) ) # create child boards
        
    new_gen = mutate_gen(new_gen, n_mutations) # mutate the unsolved boards

    new_gen.extend(solved) #re-add the solved boards

    return new_gen

def gen_algo(board_size, n_generations, gen_size, n_best=100, n_mutations=1 ): 
    """Main Method that runs the genetic algorithm
    
    Parameters
    ----------
    board_size : int
        Size of the board, N value for N-Queens
    n_generations : int
        Number of Generations to test on
    gen_size : int
        size of each generation
    n_best : int
        Number of boards to keep from each generation
    n_mutations : int
        Number of features mutations to occur for each board during the mutation process 
        
    Returns
    -------
    list
        The best board found during runtime which is a list of queen row positions. 
    """
    print("\nGoal Score: ", board_size, "\n") # Prints out the goal score for boards

    parent_gen = [] # main list of boards
    for _ in range(gen_size):
        parent_gen.append(gen_board(board_size)) # Generates and adds boards to parent_gen

    for gen in range(n_generations): # loops through the number of generations
        print(f"Best Boards in Generation {gen+1}:") #prints generation number
        print(repr( get_best(parent_gen, n_best=20, with_score=True) )) # Output best boards
        print("\n")
        parent_gen = get_next_gen(parent_gen, gen_size, n_best, n_mutations) # Sets next generation of boards

    best_board = get_best(parent_gen, n_best = 1, with_score=True) # get best board found from the algorithm with score for display
    best_board = best_board[0] # Remove from list
    print("Best Board Found:")
    print(repr(best_board)) # Print Best board found
    print("\nBoard Visualization:")
    board_2D = convert_board_to2D(best_board[0])
    print_grid(board_2D)
    print("\n")
    return best_board[0] # Return just the board, excluding the score


gen_algo(board_size=8, n_generations=20, gen_size=500, n_best=100, n_mutations=1 ) #run the genetic algorithm

