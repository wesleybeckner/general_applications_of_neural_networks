<a href="https://colab.research.google.com/github/wesleybeckner/general_applications_of_neural_networks/blob/main/notebooks/extras/X1_Tictactoe_RNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# General Applications of Neural Networks <br> X1: Reinforcement Learning Based Agents

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

---

<br>


In this lesson we'll abandon the world of heuristical agents and embrace the wilds of reinforcement learning

<br>

---


<a name='x.0'></a>

## 1.0 Preparing Environment and Importing Data

[back to top](#top)

<a name='x.0.1'></a>

### 1.0.1 Import Packages

[back to top](#top)

baselines requires an older version of TF


```python
pip install tensorflow==1.15.0
```

    Collecting tensorflow==1.15.0
      Downloading tensorflow-1.15.0-cp37-cp37m-manylinux2010_x86_64.whl (412.3 MB)
    [K     |████████████████████████████████| 412.3 MB 25 kB/s 
    [?25hCollecting keras-applications>=1.0.8
      Downloading Keras_Applications-1.0.8-py3-none-any.whl (50 kB)
    [K     |████████████████████████████████| 50 kB 6.1 MB/s 
    [?25hRequirement already satisfied: protobuf>=3.6.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (3.17.3)
    Requirement already satisfied: google-pasta>=0.1.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (0.2.0)
    Collecting tensorboard<1.16.0,>=1.15.0
      Downloading tensorboard-1.15.0-py3-none-any.whl (3.8 MB)
    [K     |████████████████████████████████| 3.8 MB 44.6 MB/s 
    [?25hCollecting tensorflow-estimator==1.15.1
      Downloading tensorflow_estimator-1.15.1-py2.py3-none-any.whl (503 kB)
    [K     |████████████████████████████████| 503 kB 55.6 MB/s 
    [?25hRequirement already satisfied: numpy<2.0,>=1.16.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (1.21.5)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (1.15.0)
    Requirement already satisfied: absl-py>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (1.0.0)
    Collecting gast==0.2.2
      Downloading gast-0.2.2.tar.gz (10 kB)
    Requirement already satisfied: wrapt>=1.11.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (1.13.3)
    Requirement already satisfied: grpcio>=1.8.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (1.43.0)
    Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (0.37.1)
    Requirement already satisfied: astor>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (0.8.1)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (3.3.0)
    Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (1.1.0)
    Requirement already satisfied: keras-preprocessing>=1.0.5 in /usr/local/lib/python3.7/dist-packages (from tensorflow==1.15.0) (1.1.2)
    Requirement already satisfied: h5py in /usr/local/lib/python3.7/dist-packages (from keras-applications>=1.0.8->tensorflow==1.15.0) (3.1.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.0) (3.3.6)
    Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.0) (57.4.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.0) (1.0.1)
    Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.0) (4.11.1)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.0) (3.7.0)
    Requirement already satisfied: typing-extensions>=3.6.4 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<1.16.0,>=1.15.0->tensorflow==1.15.0) (3.10.0.2)
    Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py->keras-applications>=1.0.8->tensorflow==1.15.0) (1.5.2)
    Building wheels for collected packages: gast
      Building wheel for gast (setup.py) ... [?25l[?25hdone
      Created wheel for gast: filename=gast-0.2.2-py3-none-any.whl size=7554 sha256=e3a2fe22299f0c6d241d9b04ee068f2d4963266bcdd239d3b3e261e433003838
      Stored in directory: /root/.cache/pip/wheels/21/7f/02/420f32a803f7d0967b48dd823da3f558c5166991bfd204eef3
    Successfully built gast
    Installing collected packages: tensorflow-estimator, tensorboard, keras-applications, gast, tensorflow
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 2.8.0
        Uninstalling tensorflow-estimator-2.8.0:
          Successfully uninstalled tensorflow-estimator-2.8.0
      Attempting uninstall: tensorboard
        Found existing installation: tensorboard 2.8.0
        Uninstalling tensorboard-2.8.0:
          Successfully uninstalled tensorboard-2.8.0
      Attempting uninstall: gast
        Found existing installation: gast 0.5.3
        Uninstalling gast-0.5.3:
          Successfully uninstalled gast-0.5.3
      Attempting uninstall: tensorflow
        Found existing installation: tensorflow 2.8.0
        Uninstalling tensorflow-2.8.0:
          Successfully uninstalled tensorflow-2.8.0
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow-probability 0.16.0 requires gast>=0.3.2, but you have gast 0.2.2 which is incompatible.
    kapre 0.3.7 requires tensorflow>=2.0.0, but you have tensorflow 1.15.0 which is incompatible.[0m
    Successfully installed gast-0.2.2 keras-applications-1.0.8 tensorboard-1.15.0 tensorflow-1.15.0 tensorflow-estimator-1.15.1


install baselines from openAI


```python
!apt-get update
!apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev
!pip install "stable-baselines[mpi]==2.9.0"
```

    Get:1 https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/ InRelease [3,626 B]
    Ign:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
    Get:3 http://security.ubuntu.com/ubuntu bionic-security InRelease [88.7 kB]    
    Ign:4 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  InRelease
    Get:5 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release [696 B]
    Hit:6 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release
    Get:7 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release.gpg [836 B]
    Get:8 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic InRelease [15.9 kB]
    Hit:9 http://archive.ubuntu.com/ubuntu bionic InRelease
    Get:10 http://archive.ubuntu.com/ubuntu bionic-updates InRelease [88.7 kB]
    Hit:11 http://ppa.launchpad.net/cran/libgit2/ubuntu bionic InRelease
    Get:13 http://archive.ubuntu.com/ubuntu bionic-backports InRelease [74.6 kB]
    Get:14 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Packages [930 kB]
    Hit:15 http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic InRelease
    Get:16 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease [21.3 kB]
    Get:17 http://security.ubuntu.com/ubuntu bionic-security/restricted amd64 Packages [806 kB]
    Get:18 http://security.ubuntu.com/ubuntu bionic-security/universe amd64 Packages [1,474 kB]
    Get:19 http://security.ubuntu.com/ubuntu bionic-security/main amd64 Packages [2,596 kB]
    Get:20 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic/main Sources [1,827 kB]
    Get:21 http://archive.ubuntu.com/ubuntu bionic-updates/main amd64 Packages [3,035 kB]
    Get:22 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic/main amd64 Packages [937 kB]
    Get:23 http://archive.ubuntu.com/ubuntu bionic-updates/universe amd64 Packages [2,252 kB]
    Get:24 http://archive.ubuntu.com/ubuntu bionic-updates/restricted amd64 Packages [840 kB]
    Get:25 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic/main amd64 Packages [42.8 kB]
    Fetched 15.0 MB in 5s (2,992 kB/s)
    Reading package lists... Done
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    zlib1g-dev is already the newest version (1:1.2.11.dfsg-0ubuntu2).
    zlib1g-dev set to manually installed.
    libopenmpi-dev is already the newest version (2.1.1-8).
    cmake is already the newest version (3.10.2-1ubuntu2.18.04.2).
    python3-dev is already the newest version (3.6.7-1~18.04).
    python3-dev set to manually installed.
    The following package was automatically installed and is no longer required:
      libnvidia-common-470
    Use 'apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 67 not upgraded.
    Collecting stable-baselines[mpi]==2.9.0
      Downloading stable_baselines-2.9.0-py3-none-any.whl (232 kB)
    [K     |████████████████████████████████| 232 kB 7.3 MB/s 
    [?25hRequirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.9.0) (3.2.2)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.9.0) (1.3.5)
    Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.9.0) (1.4.1)
    Requirement already satisfied: opencv-python in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.9.0) (4.1.2.30)
    Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.9.0) (1.1.0)
    Requirement already satisfied: gym[atari,classic_control]>=0.10.9 in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.9.0) (0.17.3)
    Requirement already satisfied: cloudpickle>=0.5.5 in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.9.0) (1.3.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from stable-baselines[mpi]==2.9.0) (1.21.5)
    Collecting mpi4py
      Downloading mpi4py-3.1.3.tar.gz (2.5 MB)
    [K     |████████████████████████████████| 2.5 MB 17.1 MB/s 
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
        Preparing wheel metadata ... [?25l[?25hdone
    Requirement already satisfied: pyglet<=1.5.0,>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from gym[atari,classic_control]>=0.10.9->stable-baselines[mpi]==2.9.0) (1.5.0)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from gym[atari,classic_control]>=0.10.9->stable-baselines[mpi]==2.9.0) (7.1.2)
    Requirement already satisfied: atari-py~=0.2.0 in /usr/local/lib/python3.7/dist-packages (from gym[atari,classic_control]>=0.10.9->stable-baselines[mpi]==2.9.0) (0.2.9)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from atari-py~=0.2.0->gym[atari,classic_control]>=0.10.9->stable-baselines[mpi]==2.9.0) (1.15.0)
    Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from pyglet<=1.5.0,>=1.4.0->gym[atari,classic_control]>=0.10.9->stable-baselines[mpi]==2.9.0) (0.16.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->stable-baselines[mpi]==2.9.0) (3.0.7)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->stable-baselines[mpi]==2.9.0) (0.11.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->stable-baselines[mpi]==2.9.0) (1.3.2)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->stable-baselines[mpi]==2.9.0) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->stable-baselines[mpi]==2.9.0) (2018.9)
    Building wheels for collected packages: mpi4py
      Building wheel for mpi4py (PEP 517) ... [?25l[?25hdone
      Created wheel for mpi4py: filename=mpi4py-3.1.3-cp37-cp37m-linux_x86_64.whl size=2185292 sha256=201eb8ee46e6f84a098ad3d83af5413f76be954aab911419894bec8346068c61
      Stored in directory: /root/.cache/pip/wheels/7a/07/14/6a0c63fa2c6e473c6edc40985b7d89f05c61ff25ee7f0ad9ac
    Successfully built mpi4py
    Installing collected packages: stable-baselines, mpi4py
    Successfully installed mpi4py-3.1.3 stable-baselines-2.9.0



```python
# Check version of tensorflow
import tensorflow as tf
tf.__version__
```




    '1.15.0'




```python
from gym import spaces
import gym
from stable_baselines.common.env_checker import check_env
```

    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    



```python
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
```

### 1.0.2 Run Tests


```python
def test_n_step_ai():
  random.seed(42)
  game = GameEngine(setup='auto', user_ai=n_step_ai)
  game.setup_game()
  game.play_game()
  # check that the winner is X
  assert game.winner == 'X', "Winner should be X!"

  # check that the ai level is set to 3 which means our engine is properly
  # accessing the user defined ai
  assert game.ai_level == 3, "The engine is not using the user defined AI!"
```


```python
test_n_step_ai()
```

    | | | |
    | | | |
    | | | |
    
    |X| | |
    | | | |
    | | | |
    
    |X| | |
    | | |O|
    | | | |
    
    |X| |X|
    | | |O|
    | | | |
    
    |X|O|X|
    | | |O|
    | | | |
    
    |X|O|X|
    | |X|O|
    | | | |
    
    |X|O|X|
    |O|X|O|
    | | | |
    
    'X' Won!
    |X|O|X|
    |O|X|O|
    | | |X|
    


## 1.1 Reinforcement Learning: Reset, Step, and Reward

Firstly, to interact with OpenAI Gym, we need to include a method of reseting the current game.

### 1.1.2 Reset


```python
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
```

Let's test our reset:


```python
game = GameEngine('auto')
game.setup_game()
game.play_game()
```

    | | | |
    | | | |
    | | | |
    
    | | | |
    | |X| |
    | | | |
    
    | | | |
    | |X| |
    | | |O|
    
    |X| | |
    | |X| |
    | | |O|
    
    |X| |O|
    | |X| |
    | | |O|
    
    |X| |O|
    | |X|X|
    | | |O|
    
    |X| |O|
    |O|X|X|
    | | |O|
    
    |X| |O|
    |O|X|X|
    |X| |O|
    
    |X|O|O|
    |O|X|X|
    |X| |O|
    
    It's a stalemate!
    |X|O|O|
    |O|X|X|
    |X|X|O|
    





    <__main__.GameEngine at 0x7f8d4b163bd0>




```python
game.reset_game()
game.play_game()
```

    | | | |
    | | | |
    | | | |
    
    | | | |
    | |X| |
    | | | |
    
    | | |O|
    | |X| |
    | | | |
    
    | | |O|
    | |X| |
    | | |X|
    
    |O| |O|
    | |X| |
    | | |X|
    
    |O|X|O|
    | |X| |
    | | |X|
    
    |O|X|O|
    | |X| |
    | |O|X|
    
    |O|X|O|
    | |X| |
    |X|O|X|
    
    |O|X|O|
    |O|X| |
    |X|O|X|
    
    It's a stalemate!
    |O|X|O|
    |O|X|X|
    |X|O|X|
    





    <__main__.GameEngine at 0x7f8d4b163bd0>



This `reset_game` function works the way we intend. However, the big step we will have to make from our current tic tac toe module to one usable by OpenAI is to work with integers rather than strings in our board representation.

### 1.1.3 Observation and Action Spaces

The following are the important changes we will have to make to our game class in order to work with OpenAI's built-in reinforcement learning algorithms:

```
# the board now has integers as values instead of strings
self.board = {1: 0,
      2: 0,
      3: 0,
      4: 0,
      5: 0,
      6: 0,
      7: 0,
      8: 0,
      9: 0,}

# the available token spaces, note that in order to access our board
# dictionary these actions will need to be re-indexed to 1
self.action_space = spaces.Discrete(9)

# the observation space requires int rep for player tokens
self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=np.int)
self.reward_range = (-10, 1)

# we will redefine our player labels as ints
self.player_label = 1
self.opponent_label = 2
```

Let's take a look at our redefined action space:


```python
board = {1: 0,
      2: 0,
      3: 0,
      4: 0,
      5: 0,
      6: 0,
      7: 0,
      8: 0,
      9: 0,}
state = np.array(list(board.values())).reshape(9,)
state
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0])



Does this align with a random sample of the observation space? It should if it is going to work!


```python
box = spaces.Box(low=0, high=2, shape=(9,), dtype=int)
box.sample()
```




    array([2, 2, 1, 2, 2, 1, 0, 1, 2])



Let's break this down. For 1 of 9 spaces (defined by shape in `spaces.Box`), the game board can take on the value of 0, 1, or 2 (defined by low and high in `spaces.Box`). When we sample from box we get a random snapshot of how the bored could possibly look. The way we've defined `state` is such that it too, represents how the board could possibly look. `state` will be returned by both `reset` and `step` when we go to wrap all of this in our game environment.

### 1.1.4 Step-Reward

Our Reinforcement Learning (RL) agent will have much less information available to them than our prior algorithms. For this we need to define our reward system a little differently. Given a current board the agent receives:

* +10 for playing a winning move
* -100 for playing an invalid move 
* -10 if the opponent wins the next move
* 1/9 for playing a valid move


```python
class TicTacToeGym(GameEngine, gym.Env):
  def __init__(self, user_ai=None, ai_level=1):
    super().__init__()
    self.setup = 'auto'
    # the default behavior will be no user_ai and ai_level set to 1 (random)
    self.user_ai = user_ai
    self.ai_level = ai_level

    # the board now has integers as values instead of strings
    self.board = {1: 0,
         2: 0,
         3: 0,
         4: 0,
         5: 0,
         6: 0,
         7: 0,
         8: 0,
         9: 0,}
    
    # the available token spaces, note that in order to access our board
    # dictionary these actions will need to be re-indexed to 1
    self.action_space = spaces.Discrete(9)

    # the observation space requires int rep for player tokens
    self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=int)
    self.reward_range = (-10, 1)

    # we will redefine our player labels as ints
    self.player_label = 1
    self.opponent_label = 2

    # for StableBaselines
    self.spec = None
    self.metadata = None

  ##############################################################################
  # we will have to redefine any function in our previous module that makes use
  # of the string entries, X and O on the board. We need to replace the logic
  # with 1's and 2's
  ##############################################################################
  def check_winning(self):
    for pattern in self.win_patterns:
      values = [self.board[i] for i in pattern] 
      if values == [1, 1, 1]:
        self.winner = 'X' # we update the winner status
        return "'X' Won!"
      elif values == [2, 2, 2]:
        self.winner = 'O'
        return "'O' Won!"
    return ''

  def check_stalemate(self):
    if (0 not in self.board.values()) and (self.check_winning() == ''):
      self.winner = 'Stalemate'
      return "It's a stalemate!"

  def reset_game(self):
    overwrite_ai = self.ai_level
    self.board = {1: 0,
         2: 0,
         3: 0,
         4: 0,
         5: 0,
         6: 0,
         7: 0,
         8: 0,
         9: 0,}
    self.winner = ''
    self.setup_game()
    self.ai_level = overwrite_ai
    # depending now on if X or O is first will need to take the AI's first step
    if self.start_player == 'O':
      move = self.random_ai()
      self.board[move] = 2

  def reset(self):
    self.reset_game()
    state = np.array(list(self.board.values())).reshape(9,)
    return state

  def random_ai(self):
    while True:
      move = random.randint(1,9)
      if self.board[move] != 0:
        continue
      else:
        break
    return move

  ##############################################################################
  # we will have to recycle a lot of what was previously wrapped up in 
  # play_game() since gym needs access to every point after the Reinf AI
  # makes a move
  ##############################################################################
  def step(self, action):

      # gym discrete indexes at 0, our board indexes at 1
      move = action + 1
      # Check if agent's move is valid
      avail_moves = [i for i in self.board.keys() if self.board[i] == 0]
      is_valid = move in avail_moves

      # if valid, then play the move, and let the other opponent make a move
      # as well
      if is_valid: # Play the move
          # update board
          self.board[move] = self.player_label
          self.check_winning()
          self.check_stalemate()

          if self.winner == '':
            ##################################################################
            # instead of continuing as we did in our play_game loop we will
            # take one additional step for the AI and then let openAI gym
            # handle incrementing between steps.
            ##################################################################

            ##########
            # Our level 1 ai agent (random)
            ##########
            # if self.ai_level == 1:
            move = self.random_ai()

            # ##########
            # # Our level 2 ai agent (heuristic)
            # ##########
            # elif self.ai_level == 2:
            #   move = self.heuristic_ai('O')

            # ##########
            # # Our user-defined AI agent
            # ##########
            # elif self.ai_level == 3:
            #   move = self.user_ai(self.board, self.win_patterns, 'O')

            self.board[move] = self.opponent_label
            self.check_winning()
            self.check_stalemate()

            if self.winner == '':
              reward, done, info = 1/9, False, {}
          
          if self.winner == 'Stalemate':
            reward, done, info = -1, True, {}

          elif self.winner == 'X':
            reward, done, info = 100, True, {}

          elif self.winner == 'O':
            reward, done, info = -10, True, {}

      else: # End the game and penalize agent
          reward, done, info = -100, True, {}

      state = np.array(list(self.board.values())).reshape(9,)
      return state, reward, done, info
```

### 1.1.5 Testing the Environment

We can check that the environment is compatible with gym using `check_env`. Notice the below doesn't return any error messages. This means everything is working ok!


```python
env = TicTacToeGym()
check_env(env)
```

We can also define a model from OpenAI and see how our game board updates in a single step with the new wrapper


```python
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = TicTacToeGym()
model = PPO2(MlpPolicy, env, verbose=1)
```

    Wrapping the env in a DummyVecEnv.
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/common/tf_util.py:58: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/common/tf_util.py:67: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/common/policies.py:115: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/common/input.py:25: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/common/policies.py:560: flatten (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.flatten instead.
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow_core/python/layers/core.py:332: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/a2c/utils.py:156: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/common/distributions.py:326: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/common/distributions.py:327: The name tf.log is deprecated. Please use tf.math.log instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/ppo2/ppo2.py:194: The name tf.summary.scalar is deprecated. Please use tf.compat.v1.summary.scalar instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/ppo2/ppo2.py:202: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/ppo2/ppo2.py:210: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/ppo2/ppo2.py:244: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/stable_baselines/ppo2/ppo2.py:246: The name tf.summary.merge_all is deprecated. Please use tf.compat.v1.summary.merge_all instead.
    



```python
# test our reset function
obs = env.reset()

# the start player should randomly select between X and O
print('the start player: {}'.format(env.start_player))

# we should return an action from model.predict
action, _states = model.predict(obs)
print("the taken action: {}".format(action))


# we divert default behavior of setup_game by saving and reestablishing our
# user input ai_level
print("AI level: {}".format(env.ai_level))

# check the board update from env.step()
obs, rewards, dones, info = env.step(action)
print(obs)

print("Should be blank if no winner: [{}]".format(env.check_winning()))
```

    the start player: O
    the taken action: 7
    AI level: 1
    [0 0 0 0 2 2 0 1 0]
    Should be blank if no winner: []


And we can still visualize the board:


```python
env.visualize_board()
```

    |0|0|0|
    |0|2|2|
    |0|1|0|
    


And check that our untrained model will win approx half the time:


```python
winners = []
for j in range(1000):
  obs = env.reset()
  for i in range(10):
      action, _states = model.predict(obs)
      # print(action)
      obs, rewards, dones, info = env.step(action)
      # env.visualize_board()
      if env.winner != '':
        winners.append(env.winner)
        break

pd.DataFrame(winners).value_counts()
```




    O            385
    X            322
    Stalemate     61
    dtype: int64



### 1.1.6 Training the Model

Now we will train the PPO2 model on our environment!


```python
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = TicTacToeGym()
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)
```


```python
obs = env.reset()
for i in range(10):
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    env.visualize_board()
    if env.winner != '':
      print(env.winner)
      break
```

    4
    |0|0|0|
    |0|1|2|
    |0|0|0|
    
    8
    |2|0|0|
    |0|1|2|
    |0|0|1|
    
    6
    |2|0|2|
    |0|1|2|
    |1|0|1|
    
    7
    |2|0|2|
    |0|1|2|
    |1|1|1|
    
    X



```python
winners = []
for j in range(1000):
  obs = env.reset()
  for i in range(10):
      action, _states = model.predict(obs)
      # print(action)
      obs, rewards, dones, info = env.step(action)
      # env.visualize_board()
      if env.winner != '':
        winners.append(env.winner)
        break
```

Let's see how many times our trained model won:


```python
pd.DataFrame(winners).value_counts()
```




    X            795
    O            172
    Stalemate     27
    dtype: int64



Not terrible! Could be better! Let's play against our model

### 1.1.7 Play Against the Model

To make our model compatible with the old `play_game` method, we will need a way to convert to and from int vs string representations on our board. Let's test this:


```python
value_map = {' ': 0,
             'X': 1,
             'O': 2}

board = {1: 'X',
         2: ' ',
         3: ' ',
         4: ' ',
         5: ' ',
         6: ' ',
         7: ' ',
         8: ' ',
         9: ' ',}

for key in board.keys():
  board[key] = value_map[board[key]]

board
```




    {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}



And now we can wrap it up into a new ai function:


```python
def rl_ai(board, win_patterns, player_label, model=model):
  # note that we are simply leaving win_patterns and player_label
  # here so that we can use the game engine as defined in prior
  # sessions, these inputs are ignored.
  
  ai_board = board.copy()
  value_map = {' ': 0,
             'X': 1,
             'O': 2}
  for key in ai_board.keys():
    ai_board[key] = value_map[ai_board[key]]
  
  obs = np.array(list(ai_board.values())).reshape(9,)
  action, _states = model.predict(obs)
  move = action + 1
  return move
```


```python
game = GameEngine('user', user_ai=rl_ai)
game.setup_game()
game.play_game()
```

    How many Players? (type 0, 1, or 2)1
    who will go first? (X, (AI), or O (Player))X
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |X| |
    | | | |
    
    O, what's your move?1
    |O| | |
    | |X| |
    | | | |
    
    |O| |X|
    | |X| |
    | | | |
    
    O, what's your move?7
    |O| |X|
    | |X| |
    |O| | |
    
    |O| |X|
    | |X| |
    |O| |X|
    
    O, what's your move?4
    'O' Won!
    |O| |X|
    |O|X| |
    |O| |X|
    





    <__main__.GameEngine at 0x7f8d40c4fb10>



Notice any interesting behaviors about the model?

## 1.2 Improve the Model

How can we improve this puppy? What about training the model against a smarter opponent? changing the reward values? training for longer? OR trying a different reinforcement learning model? Try any or all of these and see what works!


```python
class TicTacToeGym(GameEngine, gym.Env):
  def __init__(self, user_ai=None, ai_level=1):
    super().__init__()
    self.setup = 'auto'
    # the default behavior will be no user_ai and ai_level set to 1 (random)
    self.user_ai = user_ai
    self.ai_level = ai_level

    # the board now has integers as values instead of strings
    self.board = {1: 0,
         2: 0,
         3: 0,
         4: 0,
         5: 0,
         6: 0,
         7: 0,
         8: 0,
         9: 0,}
    
    # the available token spaces, note that in order to access our board
    # dictionary these actions will need to be re-indexed to 1
    self.action_space = spaces.Discrete(9)

    # the observation space requires int rep for player tokens
    self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=np.int)
    self.reward_range = (-10, 1)

    # we will redefine our player labels as ints
    self.player_label = 1
    self.opponent_label = 2

    # for StableBaselines
    self.spec = None
    self.metadata = None

  ##############################################################################
  # we will have to redefine any function in our previous module that makes use
  # of the string entries, X and O on the board. We need to replace the logic
  # with 1's and 2's
  ##############################################################################
  def check_winning(self):
    for pattern in self.win_patterns:
      values = [self.board[i] for i in pattern] 
      if values == [1, 1, 1]:
        self.winner = 'X' # we update the winner status
        return "'X' Won!"
      elif values == [2, 2, 2]:
        self.winner = 'O'
        return "'O' Won!"
    return ''

  def check_stalemate(self):
    if (0 not in self.board.values()) and (self.check_winning() == ''):
      self.winner = 'Stalemate'
      return "It's a stalemate!"

  def reset_game(self):
    overwrite_ai = self.ai_level
    self.board = {1: 0,
         2: 0,
         3: 0,
         4: 0,
         5: 0,
         6: 0,
         7: 0,
         8: 0,
         9: 0,}
    self.winner = ''
    self.setup_game()
    self.ai_level = overwrite_ai
    # depending now on if X or O is first will need to take the AI's first step
    if self.start_player == 'O':
      move = self.random_ai()
      self.board[move] = 2

  def reset(self):
    self.reset_game()
    state = np.array(list(self.board.values())).reshape(9,)
    return state

  def random_ai(self):
    while True:
      move = random.randint(1,9)
      if self.board[move] != 0:
        continue
      else:
        break
    return move

  def heuristic_ai(self, player_label):
    opponent = [1, 2]
    opponent.remove(player_label)
    opponent = opponent[0]

    avail_moves = [i for i in self.board.keys() if self.board[i] == 0]
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
        temp_board[move] = 0

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
          temp_board[move] = 0

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

  ##############################################################################
  # we will have to recycle a lot of what was previously wrapped up in 
  # play_game() since gym needs access to every point after the Reinf AI
  # makes a move
  ##############################################################################
  def step(self, action):

      # gym discrete indexes at 0, our board indexes at 1
      move = action + 1
      # Check if agent's move is valid
      avail_moves = [i for i in self.board.keys() if self.board[i] == 0]
      is_valid = move in avail_moves

      # if valid, then play the move, and let the other opponent make a move
      # as well
      if is_valid: # Play the move
          # update board
          self.board[move] = self.player_label
          self.check_winning()
          self.check_stalemate()

          if self.winner == '':
            ##################################################################
            # instead of continuing as we did in our play_game loop we will
            # take one additional step for the AI and then let openAI gym
            # handle incrementing between steps.
            ##################################################################

            ##########
            # Our level 1 ai agent (random)
            ##########
            if self.ai_level == 1:
              move = self.random_ai()

            # ##########
            # # Our level 2 ai agent (heuristic)
            # ##########
            elif self.ai_level == 2:
              move = self.heuristic_ai(self.player_label)

            # ##########
            # # Our user-defined AI agent
            # ##########
            # elif self.ai_level == 3:
            #   move = self.user_ai(self.board, self.win_patterns, 'O')

            self.board[move] = self.opponent_label
            self.check_winning()
            self.check_stalemate()

            if self.winner == '':
              reward, done, info = 1/9, False, {}
          
          if self.winner == 'Stalemate':
            reward, done, info = -10, True, {}

          elif self.winner == 'X':
            reward, done, info = 50, True, {}

          elif self.winner == 'O':
            reward, done, info = -50, True, {}

      else: # End the game and penalize agent
          reward, done, info = -100, True, {}

      state = np.array(list(self.board.values())).reshape(9,)
      return state, reward, done, info
```


```python
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = TicTacToeGym(ai_level=1)
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)
```


```python
winners = []
for j in range(1000):
  obs = env.reset()
  for i in range(10):
      action, _states = model.predict(obs)
      # print(action)
      obs, rewards, dones, info = env.step(action)
      # env.visualize_board()
      if env.winner != '':
        winners.append(env.winner)
        break
```


```python
pd.DataFrame(winners).value_counts()
```




    X            791
    O            191
    Stalemate     10
    dtype: int64




```python
game = GameEngine('user', user_ai=rl_ai)
game.setup_game()
game.play_game()
```

    How many Players? (type 0, 1, or 2)1
    who will go first? (X, (AI), or O (Player))X
    | | | |
    | | | |
    | | | |
    
    | | | |
    | |X| |
    | | | |
    
    O, what's your move?1
    |O| | |
    | |X| |
    | | | |
    
    |O| | |
    | |X| |
    |X| | |
    
    O, what's your move?2
    |O|O| |
    | |X| |
    |X| | |
    
    |O|O| |
    | |X|X|
    |X| | |
    
    O, what's your move?3
    'O' Won!
    |O|O|O|
    | |X|X|
    |X| | |
    





    <__main__.GameEngine at 0x7f6433286510>


