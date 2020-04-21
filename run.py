from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm


# Run dqn with Tetris
def dqn():
    env = Tetris()
    episodes = 2000  # 2000
    max_steps = None
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 1  # 512
    epochs = 1
    render_every = 1  # 50
    log_every = 1  # 50
    replay_start_size = 100 #2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    for episode in tqdm(range(episodes)):

        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # # Game
        # while not done and (not max_steps or steps < max_steps):
        #     next_states, board = env.get_next_states()  # I think it returns the all possible moves in next_state
        #     rows = []
        #     for row in board:
        #         # print(row)
        #         rows += row
        #     best_state = agent.best_state(next_states.values()) # returns agent best state or random state
        #
        #     best_action = None
        #     for action, state in next_states.items():
        #         if state == best_state:
        #             best_action = action
        #             break
        #
        #     reward, done = env.play(best_action[0], best_action[1], render=render,
        #                             render_delay=render_delay)
        #
        #     agent.add_to_memory(current_state, next_states[best_action], reward, done)
        #     current_state = next_states[best_action]
        #     steps += 1
        #
        # scores.append(env.get_game_score())

        # # Train original
        # if episode % train_every == 0:
        #     agent.train(batch_size=batch_size, epochs=epochs)

        # *k Game
        while not done and (not max_steps or steps < max_steps):

            next_states, board = env.get_next_states()  # returns the all possible moves in next_state
            rows = []

            for row in board:
                # print(row)
                rows += row

            # print("rows")
            # print(rows)
            # print("next")
            best_state = agent.best_state(next_states.values())  # returns agent best state prediction or random state

            best_action = None
            for action, state in next_states.items(): # Find the corresponding action for the desired state
                if state == best_state:
                    best_action = action
                    break
            # Reward each block placed yields 1 point. When clearing lines, the given score is
            # number_lines_cleared^2 × board_width. Losing a game subtracts 1 point.
            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)

            agent.add_to_memory(board, current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1
            # *k Train directly
            agent.train(batch_size=batch_size, epochs=epochs, board=rows, played_blocks=steps)
        scores.append(env.get_game_score())

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)


if __name__ == "__main__":
    dqn()
