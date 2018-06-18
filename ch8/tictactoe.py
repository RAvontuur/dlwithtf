"""Adapted from DeepChem Examples by Peter Eastman and Karl Leswing."""

import copy
import random
import shutil
import numpy as np
import tensorflow as tf
import deepchem as dc
from environment import TicTacToeEnvironment
from a3c import A3C


def eval_tic_tac_toe(value_weight,
                     num_epoch_rounds=1,
                     games=10**4,
                     rollouts=10**5,
                     advantage_lambda=0.98):
  """
  Returns the average reward over 10k games after 100k rollouts
  
  Parameters
  ----------
  value_weight: float

  Returns
  ------- 
  avg_rewards
  """
  env = TicTacToeEnvironment()
  model_dir = "/tmp/tictactoe"
  try:
    shutil.rmtree(model_dir)
  except:
    pass

  avg_rewards = []
  for j in range(num_epoch_rounds):
    print("Epoch round: %d" % j)
    a3c_engine = A3C(
        env,
        entropy_weight=0.01,
        value_weight=value_weight,
        model_dir=model_dir,
        advantage_lambda=advantage_lambda)

    a3c_engine.fit(rollouts, restore=True)
    rewards = []
    illegals = []
    losses = []
    draws = []
    wins = []
    for i in range(games):
        env.reset()
        reward = -float('inf')
        while not env.terminated:
            action = a3c_engine.select_action(env.state, deterministic=True)
            reward = env.step(action)

        rewards.append(reward)
        if abs(reward - TicTacToeEnvironment.ILLEGAL_MOVE_PENALTY) < 0.001:
            illegals.append(1.0)
        else:
            illegals.append(0.0)

        if abs(reward - TicTacToeEnvironment.LOSS_PENALTY) < 0.001:
            losses.append(1.0)
        else:
            losses.append(0.0)

        if abs(reward - TicTacToeEnvironment.DRAW_REWARD) < 0.001:
            draws.append(1.0)
        else:
            draws.append(0.0)

        if abs(reward - TicTacToeEnvironment.WIN_REWARD) < 0.001:
            wins.append(1.0)
        else:
            wins.append(0.0)

    print("Mean reward at round %d is %f" % (j + 1, np.mean(rewards)))
    print("Mean illegals at round %d is %f" % (j + 1, np.mean(illegals)))
    print("Mean losses at round %d is %f" % (j + 1, np.mean(losses)))
    print("Mean draws at round %d is %f" % (j + 1, np.mean(draws)))
    print("Mean wins at round %d is %f" % (j + 1, np.mean(wins)))

    avg_rewards.append({(j + 1) * rollouts: np.mean(rewards)})
  return avg_rewards


def main():
  score = eval_tic_tac_toe(value_weight=1.0, num_epoch_rounds=10,
                           advantage_lambda=0.5,
                           games=10**5, rollouts=2*10**5)
  print(score)


if __name__ == "__main__":
  main()
