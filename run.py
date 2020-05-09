# This is is the main, where parameters of the model can be set
# also which mode can it run in
# more detailed information can be found in the readme
from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
import itertools


# Run dqn with Tetris
def dqn():
    # training parameters
    env = Tetris()
    episodes = 200  # 2000
    max_steps = None
    epsilon_stop_episode = 150  # 1500
    mem_size = 200 # 20000
    discount = 0.95
    batch_size = 2  # 512
    epochs = 1
    render_every = 1  # 50
    log_every = 2  # 50
    replay_start_size = 200  # 2000
    train_every = 1
    n_neurons = [160, 160, 160, 160, 160]
    render_delay = None
    activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'linear']

    # all model names, line_logging and the logs will be save with the same time stamp and model_number
    # choosing which model to train
    # 1 full board or board state input also for the nuno_faria not sure
    # 2 CNN
    # 3 CNN merged
    # 4 Nuno Faria
    model_number = 3

    model_name = 'models/my_model-20200509-150301-h5' #demo: my_model-20200509-150301-h5 # my_model-20200504-233443-h5     my_model-20200508-193953-h5
    # Rendering false for Peregrine
    # board_state = True
    rendering = True
    fetch_old_model = True
    # if you choose the wrong model number for playing, you will get an error
    agent_play = True


    if model_number == 1:
        input_size = [1, 200]
    if model_number == 2:
        input_size= (1, 20, 10,1)
    if model_number == 3:
        input_size = (1, 20, 10, 1)
    if model_number == 4:
        input_size = [1, 4]

    agent = DQNAgent(env.get_board_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size, fetch_old_model=fetch_old_model,
                     model_name=model_name, model_number=model_number)

    time_frame = datetime.now().strftime("%Y%m%d-%H%M%S")

    open('lines_logging/' + f'linesfile-{time_frame}.txt', 'w') # creating file for the line logging
    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{time_frame}-{model_number}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    for episode in tqdm(range(episodes)):
        if model_number == 4:
            current_board = [0]*4
            env.reset()
        else:
            current_board = env.reset()
        done = False
        steps = 0

        if rendering:
            if render_every and episode % render_every == 0:
                render = True
            else:
                render = False
        else:
            render = False

        # *k Game
        while not done and (not max_steps or steps < max_steps):
            next_boards = env.get_next_boards(model_number)
            best_board = agent.best_board(next_boards.values(), agent_play, input_size,
                                          model_number)

            best_action = None

            for action, board in next_boards.items():  # Find the corresponding action for the desired board
                if board == best_board:
                    best_action = action
                    break
            # Reward each block placed yields 1 point. When clearing lines, the given score is
            # number_lines_cleared^2 × board_width. Losing a game subtracts 1 point.
            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay, time_frame=time_frame, model_number=model_number)

            if model_number == 1:
                agent.add_to_memory(list(itertools.chain.from_iterable(current_board)),
                                list(itertools.chain.from_iterable(next_boards[best_action])),
                                reward, done)
            else:
                agent.add_to_memory(current_board, next_boards[best_action], reward, done)
            current_board = next_boards[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % train_every == 0 and not agent_play:
            agent.train(batch_size=batch_size, epochs=epochs, model_number=model_number)
        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)
    # save the model
    if not agent_play:
        agent.model_save(time_frame,model_number)


if __name__ == "__main__":
    dqn()