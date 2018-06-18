"""Adapted from DeepChem Examples by Peter Eastman and Karl Leswing."""

import numpy as np
import shutil

from a3c import A3C
from environment import TicTacToeEnvironment

def test_run(a3c_engine, env, games, round):
    rewards = []
    illegals = []
    losses = []
    draws = []
    wins = []
    nprint = 1
    for i in range(games):
        env.reset()
        if i < nprint:
            print('GAME {}'.format(i))
        while not env.terminated:
            # if i < nprint:
            #     print('state: {}'.format(env.state))
            action, probabilities, value = a3c_engine.select_action(env.state, deterministic=True)
            reward = env.step(action)
            if i < nprint:
                print('action: {}'.format(action))
                print('reward: {}'.format(reward))
                print('probabilities: {}'.format(probabilities))
                print('value: {}'.format(value))
                print(env.display())

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

    print("Mean reward at round %d is %f" % (round, np.mean(rewards)))
    print("Mean illegals at round %d is %f" % (round, np.mean(illegals)))
    print("Mean losses at round %d is %f" % (round, np.mean(losses)))
    print("Mean draws at round %d is %f" % (round, np.mean(draws)))
    print("Mean wins at round %d is %f" % (round, np.mean(wins)))
    return rewards

def eval_tic_tac_toe(value_weight,
                     num_epoch_rounds=1,
                     games=10 ** 4,
                     rollouts=10 ** 5,
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

    a3c_engine = A3C(
            env,
            entropy_weight=0.01,
            value_weight=value_weight,
            model_dir=model_dir,
            advantage_lambda=advantage_lambda)
    a3c_engine.initialize()

    test_run(a3c_engine, env, games, 0)

    avg_rewards = []
    for j in range(num_epoch_rounds):
        print("Epoch round: %d" % (j+1))

        a3c_engine = A3C(
            env,
            entropy_weight=0.01,
            value_weight=value_weight,
            model_dir=model_dir,
            advantage_lambda=advantage_lambda)

        a3c_engine.fit(rollouts, restore=True)

        rewards = test_run(a3c_engine, env, games, j+1)

        avg_rewards.append({(j + 1) * rollouts: np.mean(rewards)})
    return avg_rewards


def main():
    value_weight = 6.0
    score = eval_tic_tac_toe(value_weight=10., num_epoch_rounds=20,
                             advantage_lambda=0.5,
                             games=10 ** 4, rollouts=5 * 10 ** 4)
    print(score)


if __name__ == "__main__":
    main()
