import os
import argparse

import torch

import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger, plot_curve

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    
    return agent

def main():
    # Make environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = rlcard.make('no-limit-holdem', config={'seed': 0, 'env_num': 4})
    eval_env = rlcard.make('no-limit-holdem', config={'seed': 0, 'env_num': 4})

    # Set the iterations numbers and how frequently we evaluate performance
    evaluate_every = 5000
    selfplay_every = 25000
    evaluate_num = 10000
    iteration_num = 8000000

    # The intial memory size
    memory_init_size = 100

    # Train the agent every X steps
    train_every = 1
    
    agent = DQNAgent(num_actions=env.num_actions,
                         state_shape=env.state_shape[0],
                         mlp_layers=[64,64, 64, 64],
                         device=device)
    
    agents = [agent, load_model("model.pth")]
    
    env.set_agents(agents)
    
    with Logger('./') as logger:
        for episode in range(iteration_num):

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                logger.log_performance(env.timestep, tournament(env, evaluate_num)[0])
            if episode % selfplay_every == 0:
                save_path = os.path.join('./',  str(episode) + "model.pth")
                torch.save(agent, save_path)
                print('Model saved in', save_path)
                agents = [agent, load_model(str(episode) + "model.pth")]
                env.set_agents(agents)
                

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    #plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join('./', 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)
    
    

    # The paths for saving the logs and learning curves
    log_dir = './experiments/nlh_cfr_result/'

    # Set a global seed
    set_seed(0)
    
if __name__ == '__main__':
    main()