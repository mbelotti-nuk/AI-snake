import numpy as np
from utils.agent import Agent
from utils.snake_game import SnakeGame

    
# number of training episodes
MAX_EPISODES = 4500
# episode from which flat epsilon greedy is considered (i.e. epsilon=epsilon_min=0.1)
NUM_EPISODE_FLAT_EPS = 3500
# number of episodes to be observed
GAMES_OBSERVE = 400


# Create an agent
agent = Agent(device="mps", lr=1E-4, batch_size=512)
agent.define_epsilon_startegy(num_games_end=NUM_EPISODE_FLAT_EPS, num_games_min=GAMES_OBSERVE)
game = SnakeGame()

scores = []
max_score = 0


for episode in range(0, MAX_EPISODES):
    
    done = False
    # re-set the environment
    game.reset()

    score = 0
    frame_count = 0
    

    while not done:

        # Take epsilon greedy action
        state = game.state 
        action = agent.choose_action(state)

        # make a step
        next_state, reward, done, score, action = game.step(action)

        # Preprocess next state and store transition
        agent.store_experience(state, next_state, action, reward, done) 
        
        if episode > GAMES_OBSERVE:
            agent.learn()

        frame_count = frame_count + 1

    agent.update_num_episodes()

    # Maintain record of the max score achieved so far
    if score > max_score:
        max_score = score

    mean_score = np.mean(scores[-100:])

    scores.append(score)


    if(episode % 10 == 0):
        print(f'Episode {episode}: \n\tScore: {score}\n\tAvg score (past 100): {mean_score}\
                \n\tEpsilon: {agent.eps}\n')

            

print("save model")
agent.policy_net.save_model()
agent.plot_results(scores)