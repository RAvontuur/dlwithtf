"""Adapted from DeepChem Examples by Peter Eastman and Karl Leswing."""

import copy
import random
import shutil
import numpy as np
import tensorflow as tf
import deepchem as dc
from environment import TicTacToeEnvironment
from a3c import A3C


def eval_tic_tac_toe():
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
  num_epoch_rounds=10
  for j in range(num_epoch_rounds):
    print("Epoch round: %d" % j)

    learning_rates=[0.01,0.01,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    if j < 4:
        games=10**3
        rollouts=2*10**4
        train_rules=True
        value_weight=1.0
        entropy_weight=0.0
        advantage_lambda=0.0
        discount_factor=0.0
        env.reward_rules_only(True)
        random_train=True
    else:
        games=10**4
        rollouts=2*10**5
        train_rules=False
        value_weight=1.0
        entropy_weight=0.01
        advantage_lambda=0.5
        discount_factor=0.5
        env.reward_rules_only(False)
        random_train=False

    a3c_engine = A3C(
        env,
        entropy_weight=entropy_weight,
        discount_factor=discount_factor,
        value_weight=value_weight,
        model_dir=model_dir,
        learning_rate=learning_rates[j],
        advantage_lambda=advantage_lambda,
        train_rules= train_rules,
        random_train = random_train
        )

    a3c_engine.fit(rollouts, restore=True)

    # validation run
    # env.reward_rules_only(False)
    rewards = []
    illegals = []
    losses = []
    draws = []
    wins = []
    for i in range(games):
        env.reset()
        reward = -float('inf')
        while not env.terminated:
            action, probabilities, value = a3c_engine.select_action(env.state, deterministic=True)
            reward = env.step(action)
           # if i < 10:
                # print('action: {}'.format(action))
                # print('reward: {}'.format(reward))
                # print('probabilities: {}'.format(probabilities))
                # print('value: {}'.format(value))
                # print(env.display())

        rewards.append(reward)
        if abs(reward - env.ILLEGAL_MOVE_PENALTY) < 0.001:
            illegals.append(1.0)
        else:
            illegals.append(0.0)

        if abs(reward - env.LOSS_PENALTY) < 0.001:
            losses.append(1.0)
        else:
            losses.append(0.0)

        if abs(reward - env.DRAW_REWARD) < 0.001:
            draws.append(1.0)
        else:
            draws.append(0.0)

        if abs(reward - env.WIN_REWARD) < 0.001:
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
  score = eval_tic_tac_toe()
  print(score)


if __name__ == "__main__":
  main()
