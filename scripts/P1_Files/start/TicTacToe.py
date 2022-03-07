import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def n_step_ai(board, win_patterns, player_label, n_steps=3):
  opponent = ['X', 'O']
  opponent.remove(player_label)
  opponent = opponent[0]

  avail_moves = {i: 1 for i in board.keys() if board[i] == ' '}
  
  for move in avail_moves.keys():
    temp_board = board.copy()
    temp_board[move] = player_label
    score = get_minimax(n_steps, temp_board, player_label)
    avail_moves[move] = score

  ##########################################
  ### The rest of our ai agent harness is the same
  ##########################################

  # first grab max score
  max_score = max(avail_moves.values())

  # then select all moves that have this max score
  valid = []
  for key, value in avail_moves.items():
    if value == max_score:
      valid.append(key)

  # return a random selection of the moves with the max score
  move = random.choice(valid)

  return move

def minimax(depth, board, maximizing_player, player_label, verbiose=False):
  # infer the opponent
  opponent = ['X', 'O']
  opponent.remove(player_label)
  opponent = opponent[0]

  # set the available moves
  avail_moves = [i for i in board.keys() if board[i] == ' ']

  # check if the depth is 0, or stalemate/winner has been reached
  # if so this is the basecase and we want to return get_score()
  terminal_move = is_terminal_node(board, avail_moves)

  if terminal_move or depth == 0:
    score = get_score(board, player_label, win_patterns)
    if verbiose:
      print('{} score: {}. depth: {}'.format(board, score, depth))
    return score
  
  ### in the following we want to search through every possible board at the 
  ### current level (the possible moves for the current player, given that the
  ### player is either the one whose turn it is or the imagined opponent)

  # call minimax where it is the current players turn and so we want to 
  # maximize the score
  if maximizing_player:
    score = -np.Inf
    for move in avail_moves:
      new_board = board.copy()
      new_board[move] = player_label
      score = max(score, minimax(depth-1, new_board, False, player_label, verbiose))
    if verbiose:
      print('{} max. score: {}. depth: {}'.format(board, score, depth))
    return score

  # call minimax where it is the opponent players turn and so we want to
  # minimize the score
  elif not maximizing_player:
    score = np.Inf
    for move in avail_moves:
      new_board = board.copy()
      new_board[move] = opponent
      score = min(score, minimax(depth-1, new_board, True, player_label, verbiose))
    if verbiose:
      print('{} min. score: {}. depth: {}'.format(board, score, depth))
    return score

def is_terminal_node(board, avail_moves):
  if check_winning(board, win_patterns):
    return True
  elif check_stalemate(board, win_patterns):
    return True
  else:
    return False

def get_score(board, player_label, win_patterns):
  # this will look somewhat similar to our 1-step lookahead algorithm
  opponent = ['X', 'O']
  opponent.remove(player_label)
  opponent = opponent[0]
  score = 0
  for pattern in win_patterns:
      values = [board[i] for i in pattern] 
      # if the opponent wins, the score is -100
      if values == [opponent, opponent, opponent]:
        score = -100
      elif values == [player_label, player_label, player_label]:
        score = 100
  return score

# we're going to pull out and reformat some of our helper functions in the
# TicTacToe class

win_patterns = [[1,2,3], [4,5,6], [7,8,9],
                [1,4,7], [2,5,8], [3,6,9],
                [1,5,9], [7,5,3]]

def check_winning(board, win_patterns):
  for pattern in win_patterns:
    values = [board[i] for i in pattern] 
    if values == ['X', 'X', 'X'] or values == ['O', 'O', 'O']:
      return True
  return False

def check_stalemate(board, win_patterns):
  if (' ' not in board.values()) and (check_winning(board, win_patterns) == ''):
    return True
  return False

def get_minimax(depth, board, player_label, verbiose=False):
  score = minimax(depth-1, board, False, player_label, verbiose=verbiose)
  return score

def n_step_ai_temp(board, win_patterns, player_label, n_steps, verbiose=False):
  opponent = ['X', 'O']
  opponent.remove(player_label)
  opponent = opponent[0]

  avail_moves = {i: 1 for i in board.keys() if board[i] == ' '}
  
  for move in avail_moves.keys():
    temp_board = board.copy()
    temp_board[move] = player_label
    score = get_minimax(n_steps, temp_board, player_label, verbiose=verbiose)
    avail_moves[move] = score
  return avail_moves

def one_step_ai(board, win_patterns, player_label):
  opponent = ['X', 'O']
  opponent.remove(player_label)
  opponent = opponent[0]

  avail_moves = {i: 1 for i in board.keys() if board[i] == ' '}
  temp_board = board.copy()
  ########################################
  # we're going to change the following lines, instead of caring
  # whether we've found the best move, we want to update the move
  # with a score
  ########################################

  # check if the opponent has a winning move first, we will overwrite
  # the score for this move if it is also a winning move for the current 
  # player
  for move in avail_moves.keys():
    temp_board[move] = opponent
    for pattern in win_patterns:
        values = [temp_board[i] for i in pattern] 
        if values == [opponent, opponent, opponent]:
          avail_moves[move] = 10
    temp_board[move] = ' '

  for move in avail_moves.keys():
    temp_board[move] = player_label
    for pattern in win_patterns:
        values = [temp_board[i] for i in pattern] 
        if values == [player_label, player_label, player_label]:
          avail_moves[move] = 100
    temp_board[move] = ' '

  # first grab max score
  max_score = max(avail_moves.values())

  # then select all moves that have this max score
  valid = []
  for key, value in avail_moves.items():
    if value == max_score:
      valid.append(key)

  # return a random selection of the moves with the max score
  move = random.choice(valid)

  return move

class TicTacToe:
  # can preset winner and starting player
  def __init__(self, winner='', start_player=''): 
    self.winner = winner
    self.start_player = start_player
    self.board = {1: ' ',
         2: ' ',
         3: ' ',
         4: ' ',
         5: ' ',
         6: ' ',
         7: ' ',
         8: ' ',
         9: ' ',}
    self.win_patterns = [[1,2,3], [4,5,6], [7,8,9],
                [1,4,7], [2,5,8], [3,6,9],
                [1,5,9], [7,5,3]]
         
  # the other functions are now passed self
  def visualize_board(self):
    print(
      "|{}|{}|{}|\n|{}|{}|{}|\n|{}|{}|{}|\n".format(*self.board.values())
      )

  def check_winning(self):
    for pattern in self.win_patterns:
      values = [self.board[i] for i in pattern] 
      if values == ['X', 'X', 'X']:
        self.winner = 'X' # we update the winner status
        return "'X' Won!"
      elif values == ['O', 'O', 'O']:
        self.winner = 'O'
        return "'O' Won!"
    return ''

  def check_stalemate(self):
    if (' ' not in self.board.values()) and (self.check_winning() == ''):
      self.winner = 'Stalemate'
      return "It's a stalemate!"

class GameEngine(TicTacToe):
  def __init__(self, setup='auto', user_ai=None):
    super().__init__()
    self.setup = setup
    self.user_ai = user_ai

  def heuristic_ai(self, player_label):
    opponent = ['X', 'O']
    opponent.remove(player_label)
    opponent = opponent[0]

    avail_moves = [i for i in self.board.keys() if self.board[i] == ' ']
    temp_board = self.board.copy()
    middle = 5
    corner = [1,3,7,9]
    side = [2,4,6,8]

    # first check for a winning move
    move_found = False
    for move in avail_moves:
      temp_board[move] = player_label
      for pattern in self.win_patterns:
          values = [temp_board[i] for i in pattern] 
          if values == [player_label, player_label, player_label]:
            move_found = True       
            break
      if move_found:   
        break
      else:
        temp_board[move] = ' '

    # check if the opponent has a winning move
    if move_found == False:
      for move in avail_moves:
        temp_board[move] = opponent
        for pattern in self.win_patterns:
            values = [temp_board[i] for i in pattern] 
            if values == [opponent, opponent, opponent]:
              move_found = True       
              break
        if move_found:   
          break
        else:
          temp_board[move] = ' '

    # check if middle avail
    if move_found == False:
      if middle in avail_moves:
        move_found = True
        move = middle

    # check corners
    if move_found == False:
      move_corner = [val for val in avail_moves if val in corner]
      if len(move_corner) > 0:
        move = random.choice(move_corner)
        move_found = True

    # check side
    if move_found == False:
      move_side = [val for val in avail_moves if val in side]
      if len(move_side) > 0:
        move = random.choice(move_side)
        move_found = True

    return move

  def random_ai(self):
    while True:
      move = random.randint(1,9)
      if self.board[move] != ' ':
        continue
      else:
        break
    return move

  def setup_game(self):
    if self.setup == 'user':
      players = int(input("How many Players? (type 0, 1, or 2)"))
      self.player_meta = {'first': {'label': 'X',
                                    'type': 'ai'}, 
                    'second': {'label': 'O',
                                    'type': 'human'}}
      if players != 2:
        ########## 
        # Allow the user to set the ai level
        ########## 

        ### if they have not provided an ai_agent
        if self.user_ai == None:
          level = int(input("select AI level (1, 2)"))
          if level == 1:
            self.ai_level = 1
          elif level == 2:
            self.ai_level = 2
          else:
            print("Unknown AI level entered, this will cause problems")
        else:
          self.ai_level = 3

      if players == 1:
        first = input("who will go first? (X, (AI), or O (Player))")
        if first == 'O':
          self.player_meta = {'second': {'label': 'X',
                                    'type': 'ai'}, 
                        'first': {'label': 'O',
                                    'type': 'human'}}
        

      elif players == 0:
        first = random.choice(['X', 'O'])
        if first == 'O':
          self.player_meta = {'second': {'label': 'X',
                                    'type': 'ai'}, 
                        'first': {'label': 'O',
                                    'type': 'ai'}}                                
        else:
          self.player_meta = {'first': {'label': 'X',
                                    'type': 'ai'}, 
                        'second': {'label': 'O',
                                    'type': 'ai'}}

        
    elif self.setup == 'auto':
      first = random.choice(['X', 'O'])
      if first == 'O':
        self.start_player = 'O'
        self.player_meta = {'second': {'label': 'X',
                                  'type': 'ai'}, 
                      'first': {'label': 'O',
                                  'type': 'ai'}}                                
      else:
        self.start_player = 'X'
        self.player_meta = {'first': {'label': 'X',
                                  'type': 'ai'}, 
                      'second': {'label': 'O',
                                  'type': 'ai'}}
      ########## 
      # and automatically set the ai level otherwise
      ##########  
      if self.user_ai == None:                           
        self.ai_level = 2
      else:
        self.ai_level = 3
  
  def make_move(self, move=None, player='first'):
    """
    player will be either 'first' or 'second'
    meta data will then infer game play.

    Will return the board (to display), potentially a message
    to display as well.

    the purpose of this function is to integrate with a larger
    application (ie flask) where it wouldn't suffice to have an internal 
    loop (ie play_game function)

    returns the winner (stalemate, X, O or None) and the board
    """

    ### first player
    player_label = self.player_meta[player]['label']
    player_type = self.player_meta[player]['type']

    self.board[move] = player_label
    self.check_winning()
    self.check_stalemate()

    if self.winner == 'Stalemate':
      print(self.check_stalemate())
      self.visualize_board()
      return 'Stalemate', self.board

    elif self.winner != '':
      print(self.check_winning())
      self.visualize_board()
      return self.winner, self.board

    ### second player
    players = ['first', 'second']
    players.remove(player)
    player = players[0]

    player_label = self.player_meta[player]['label']
    player_type = self.player_meta[player]['type']
    if player_type == 'human':
      return None, self.board # next players move
    else:
      ##########
      # Our level 1 ai agent (random)
      ##########
      if self.ai_level == 1:
        move = self.random_ai()

      ##########
      # Our level 2 ai agent (heuristic)
      ##########
      elif self.ai_level == 2:
        move = self.heuristic_ai(player_label)

      ##########
      # Our user-defined AI agent
      ##########
      elif self.ai_level == 3:
        move = self.user_ai(self.board, self.win_patterns, player_label)

    self.board[move] = player_label

    # the winner varaible will now be check within the board object
    self.check_winning()
    self.check_stalemate()
    if self.winner == 'Stalemate':
      print(self.check_stalemate())
      self.visualize_board()
      return 'Stalemate', self.board

    elif self.winner != '':
      print(self.check_winning())
      self.visualize_board()
      return self.winner, self.board

    return None, self.board
      
  
  def play_game(self):
    while True:
      for player in ['first', 'second']:  
        self.visualize_board()
        player_label = self.player_meta[player]['label']
        player_type = self.player_meta[player]['type']

        if player_type == 'human':
          move = input("{}, what's your move?".format(player_label))
          # we're going to allow the user to quit the game from the input line
          if move in ['q', 'quit']:
            self.winner = 'F'
            print('quiting the game')
            break

          move = int(move)
          if self.board[move] != ' ':
            while True:
              move = input("{}, that position is already taken! "\
                          "What's your move?".format(player_label))  
              move = int(move)            
              if self.board[move] != ' ':
                continue
              else:
                break

        else:
          ##########
          # Our level 1 ai agent (random)
          ##########
          if self.ai_level == 1:
            move = self.random_ai()

          ##########
          # Our level 2 ai agent (heuristic)
          ##########
          elif self.ai_level == 2:
            move = self.heuristic_ai(player_label)

          ##########
          # Our user-defined AI agent
          ##########
          elif self.ai_level == 3:
            move = self.user_ai(self.board, self.win_patterns, player_label)

        self.board[move] = player_label

        # the winner varaible will now be check within the board object
        self.check_winning()
        self.check_stalemate()

        if self.winner == '':
          continue

        elif self.winner == 'Stalemate':
          print(self.check_stalemate())
          self.visualize_board()
          break

        else:
          print(self.check_winning())
          self.visualize_board()
          break
      if self.winner != '':
        return self

  ####################################
  # Adding our ability to reset the game
  ####################################
  def reset_game(self):
    self.board = {1: ' ',
         2: ' ',
         3: ' ',
         4: ' ',
         5: ' ',
         6: ' ',
         7: ' ',
         8: ' ',
         9: ' ',}
    self.winner = ''
    self.setup_game()