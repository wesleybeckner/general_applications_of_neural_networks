from flask import Flask, render_template_string, request, make_response
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

##### OLD STUFF ###

class TicTacToe(TwoPlayerGame):
    """The board positions are numbered as follows:
    1 2 3
    4 5 6
    7 8 9
    """

    def __init__(self, players):
        self.players = players
        self.board = [0 for i in range(9)]
        self.current_player = 1  # player 1 starts.

    def possible_moves(self):
        return [i + 1 for i, e in enumerate(self.board) if e == 0]

    def make_move(self, move):
        self.board[int(move) - 1] = self.current_player

    def unmake_move(self, move):  # optional method (speeds up the AI)
        self.board[int(move) - 1] = 0

    WIN_LINES = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],  # horiz.
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],  # vertical
        [1, 5, 9],
        [3, 5, 7],  # diagonal
    ]

    def lose(self, who=None):
        """ Has the opponent "three in line ?" """
        if who is None:
            who = self.opponent_index
        wins = [
            all([(self.board[c - 1] == who) for c in line]) for line in self.WIN_LINES
        ]
        return any(wins)

    def is_over(self):
        return (
            (self.possible_moves() == [])
            or self.lose()
            or self.lose(who=self.current_player)
        )

    def show(self):
        print(
            "\n"
            + "\n".join(
                [
                    " ".join([[".", "O", "X"][self.board[3 * j + i]] for i in range(3)])
                    for j in range(3)
                ]
            )
        )

    def spot_string(self, i, j):
        return ["_", "O", "X"][self.board[3 * j + i]]

    def scoring(self):
        opp_won = self.lose()
        i_won = self.lose(who=self.current_player)
        if opp_won and not i_won:
            return -100
        if i_won and not opp_won:
            return 100
        return 0

    def winner(self):
        if self.lose(who=2):
            return "AI Wins"
        return "Tie"


TEXT = """
<!doctype html>
<html>
  <head><title>Tic Tac Toe</title></head>
  <body>
    <h1>Tic Tac Toe</h1>
    <h2>{{msg}}</h2>
    <form action="" method="POST">
      <table>
        {% for j in range(0, 3) %}
        <tr>
          {% for i in range(0, 3) %}
          <td>
            <button type="submit" name="choice" value="{{j*3+i+1}}"
             {{"disabled" if ttt.spot_string(i, j)!="_"}}>
              {{ttt.spot_string(i, j)}}
            </button>
          </td>
          {% endfor %}
        </tr>
        {% endfor %}
      </table>
      <button type="submit" name="reset">Start Over</button>
    </form>
  </body>
</html>
"""

app = Flask(__name__)
ai_algo = Negamax(6)


@app.route("/", methods=["GET", "POST"])
def play_game():
    ttt = TicTacToe([Human_Player(), AI_Player(ai_algo)])
    game_cookie = request.cookies.get("game_board")
    if game_cookie:
        ttt.board = [int(x) for x in game_cookie.split(",")]
    if "choice" in request.form:
        ttt.play_move(request.form["choice"])
        if not ttt.is_over():
            ai_move = ttt.get_move()
            ttt.play_move(ai_move)
    if "reset" in request.form:
        ttt.board = [0 for i in range(9)]
    if ttt.is_over():
        msg = ttt.winner()
    else:
        msg = "play move"
    resp = make_response(render_template_string(TEXT, ttt=ttt, msg=msg))
    c = ",".join(map(str, ttt.board))
    resp.set_cookie("game_board", c)
    return resp


if __name__ == "__main__":
    app.run()
