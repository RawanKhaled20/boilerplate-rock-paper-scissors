# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import numpy as np

# Def some functions
def determine_reward(you, opponent): # 1: win; 0: tie; -1: loss
  winning_situations = [['R','S'],['S','P'],['P','R']]
  if [you, opponent] in winning_situations:
      return 10
  elif you == opponent:
      return 0
  else:
      return -10

def number_to_symbol(num_action):
  if num_action == 0:
    return 'R'
  elif num_action == 1:
    return 'P'
  else:
    return 'S'

def symbol_to_number(sym_action):
  if sym_action == 'R':
    return 0
  elif sym_action == 'P':
    return 1
  elif sym_action == 'S':
    return 2

# Initialize variables 
States = {('R', 'R'): 0,
          ('R', 'P'): 1,
          ('R', 'S'): 2,
          ('P', 'R'): 3,
          ('P', 'P'): 4,
          ('P', 'S'): 5,
          ('S', 'R'): 6,
          ('S', 'P'): 7,
          ('S', 'S'): 8}

Q = np.zeros((9, 3))
alpha = 0.25  #learning rate
gamma = 0.3   #The discount factor balances the weight of future rewards in the Q-learning update.
S = 0 #current state
n = 1 # steps for episode (each game is considered one episode)

# Q-Learning
def player(prev_opponent_play, opponent_history=[]):

  global States
  global Q
  global alpha
  global gamma
  global S
  global A

  opponent_history.append(prev_opponent_play)

  if prev_opponent_play == '':
    A = 'S'
    return A
  else:
    # n, a complete game till end
    for _ in range(n):
      R = determine_reward(A, prev_opponent_play)
      Sn = States[(A, prev_opponent_play)] # S_{t+1}
      a = int(Q[Sn,:].argmax())
      A_idx = symbol_to_number(A)
      
      Q[S, A_idx] = Q[S, A_idx] + alpha*(R + gamma*Q[Sn, a] - Q[S, A_idx])
    
      A = number_to_symbol(int(Q[Sn,:].argmax()))
      S = Sn
    return A
