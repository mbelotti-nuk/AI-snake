from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from utils.agent import Agent
from utils.snake_game import SnakeGame

def save_gif(frames):
    frames = [Image.fromarray(image, mode='RGB')for image in frames]
    frame_one = frames[0]
    frame_one.save("results/match.gif", format="GIF", append_images=frames,
                save_all=True, duration=20, loop=0)
    
NUM_EPISODES = 10


# Create an agent
agent = Agent(device="mps")
agent.load_net("models/LDQN.pt")

game = SnakeGame()

scores = []
best_game_frames = []
max_score = 0


for episode in range(0, NUM_EPISODES):
    
    done = False
    # re-set the environment
    game.reset()
    frames = []
    score = 0
    frame_count = 0
    

    while not done:

        # Take epsilon greedy action
        state = game.state 

        action = agent.choose_action(state, train=False)

        # make a step
        next_state, reward, done, score, action = game.step(action)
        frames.append(game.frame)

    # Maintain record of the max score achieved so far
    if score > max_score:
        max_score = score
        best_game_frames = frames

    mean_score = np.mean(scores[-100:])

    scores.append(score)
    print(f"Score {score}")

            
save_gif(best_game_frames)
