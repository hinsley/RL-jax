import random
random.seed(0)

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import glob
import pickle
import time
from datetime import datetime, timedelta

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Starting learning rate.
LEARNING_RATE = 1e-5
# Discount factor for future rewards.
GAMMA = 0.99
# Total number of episodes to train for.
NUM_EPISODES = 10000
# Number of episodes to average for learning rate schedule checks.
LR_SCHEDULE_AVERAGE_N = 20
# Learning rate schedule: (reward_threshold, new_learning_rate) pairs.
# LR_SCHEDULE = [
#     (50, 3e-4),
#     (100, 1e-4),
#     (200, 3e-5),
# ]
LR_SCHEDULE = []
# Whether to load existing weights or start fresh.
LOAD_EXISTING_WEIGHTS = True
# How often to save weights during training (in episodes).
SAVE_FREQUENCY = 500
# Whether to render (display) every episode during training.
RENDER_ALL_EPISODES = True
# Whether to record video of the best episode.
RECORD_BEST_EPISODE = False

class LunarLanderNetwork(hk.Module):
    """Neural network for the Lunar Lander environment.
    
    Architecture:
    - Input layer: 8 neurons (state observations)
    - Hidden layer 1: 512 neurons with bias and ReLU activation
    - Hidden layer 2: 512 neurons with bias and ReLU activation
    - Hidden layer 3: 512 neurons with bias and ReLU activation
    - Output layer: 4 neurons with softmax activation (action probabilities)
    
    LayerNorm is applied after each linear layer to normalize activations,
    which helps with training stability and convergence speed.
    """
    
    def __init__(self, output_size: int = 4):
        super().__init__(name="LunarLanderNetwork")
        self.output_size = output_size
        
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network.
        
        Args:
            observations: Input tensor with shape [batch_size, 8].
            
        Returns:
            Action probabilities with shape [batch_size, 4].
        """
        x = observations
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.output_size)(x)
        return jax.nn.softmax(x)

def lunar_lander_policy():
    """Creates the lunar lander policy network."""
    def forward(observations):
        network = LunarLanderNetwork()
        return network(observations)
    
    return forward

# Transform the policy function into a pair of pure functions.
policy = hk.transform(lunar_lander_policy())

# Initialize parameters.
def init_params(rng_key, dummy_obs):
    """Initialize the network parameters."""
    return policy.init(rng_key, dummy_obs)

# Sample actions from the policy.
def sample_action(params, rng_key, observation):
    """Sample an action from the policy distribution."""
    logits = policy.apply(params, rng_key, observation)
    # Add a small amount of exploration noise to prevent zeros.
    logits = logits + 1e-8
    return jax.random.categorical(rng_key, jnp.log(logits))

# Calculate per-step gradients for REINFORCE.
def compute_step_gradient(params, rng_key, observation, action, reward_to_go):
    """Compute the gradient contribution for a single step."""
    # Compute action probabilities.
    action_probs = policy.apply(params, rng_key, observation[None])  # Add batch dimension.
    
    # Convert action to one-hot.
    one_hot_action = jax.nn.one_hot(action, action_probs.shape[1])
    
    # Select probability of the action taken.
    selected_prob = jnp.sum(action_probs * one_hot_action)
    
    # Compute log probability.
    log_prob = jnp.log(selected_prob)
    
    # Define a function to get the gradient of log_prob with respect to params.
    def log_prob_fn(p):
        a_probs = policy.apply(p, rng_key, observation[None])
        sel_prob = jnp.sum(a_probs * one_hot_action)
        return jnp.log(sel_prob)
    
    # Compute gradient of log probability.
    grad = jax.grad(log_prob_fn)(params)
    
    # Scale gradient by reward-to-go for this step.
    scaled_grad = jax.tree.map(lambda g: g * reward_to_go, grad)
    
    # Make objective negative so it behaves like a proper loss function
    return scaled_grad, -log_prob * reward_to_go  # Added negative sign here

# Collect a single episode and compute per-step gradients.
def collect_episode_gradients(env, params, rng_key):
    """Collect a full episode and compute per-step gradients."""
    observation, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    # Lists to store episode data.
    observations = []
    actions = []
    rewards = []
    
    # Collect the episode trajectory.
    while not done:
        step_count += 1
        # Sample action.
        rng_key, action_key = jax.random.split(rng_key)
        action = sample_action(params, action_key, jnp.array(observation))
        action_int = int(action)
        
        # Take action in environment.
        next_observation, reward, terminated, truncated, _ = env.step(action_int)
        done = terminated or truncated
        total_reward += reward
        
        # Store step data.
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        
        observation = next_observation
    
    # Convert to arrays.
    observations = jnp.array(observations)
    actions = jnp.array(actions)
    rewards = jnp.array(rewards)
    
    # Compute rewards-to-go for each step.
    rewards_to_go = np.zeros_like(rewards)
    running_sum = 0
    # Calculate backwards from the last reward.
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + GAMMA * running_sum
        rewards_to_go[t] = running_sum
    
    # Normalize rewards-to-go for more stable training and meaningful loss values
    if len(rewards_to_go) > 1:
        rewards_to_go = (rewards_to_go - np.mean(rewards_to_go)) / (np.std(rewards_to_go) + 1e-8)
    
    # Initialize gradient accumulator.
    grad_accumulator = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    objective_value = 0.0
    
    # Compute and accumulate gradients for each step.
    for t in range(len(observations)):
        rng_key, grad_key = jax.random.split(rng_key)
        step_grad, step_obj = compute_step_gradient(
            params, grad_key, observations[t], actions[t], rewards_to_go[t])
        
        # Accumulate gradient.
        grad_accumulator = jax.tree.map(
            lambda acc, g: acc + g, 
            grad_accumulator, 
            step_grad
        )
        
        objective_value += step_obj
    
    return grad_accumulator, total_reward, objective_value

# Main training loop with episodic updates.
def train_reinforce(
    num_episodes=NUM_EPISODES,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    render_every=100,
    lr_schedule=LR_SCHEDULE,
    save_frequency=SAVE_FREQUENCY,
    load_existing=LOAD_EXISTING_WEIGHTS
):
    """Train the agent using REINFORCE with per-step gradient calculation."""
    env = gym.make("LunarLander-v3")
    
    # Initialize random key.
    rng_key = jax.random.PRNGKey(0)
    
    # Initialize parameters or load existing ones.
    rng_key, init_key = jax.random.split(rng_key)
    dummy_obs = jnp.zeros((8,))  # Lunar Lander has 8 observations.
    
    if load_existing:
        loaded_params = load_latest_weights()
        if loaded_params is not None:
            params = loaded_params
        else:
            params = init_params(init_key, dummy_obs)
    else:
        print("Starting with fresh weights.")
        params = init_params(init_key, dummy_obs)
    
    # Set up learning rate schedule if provided.
    if lr_schedule is None:
        lr_schedule = []  # Empty schedule means no adjustments.
    
    # Sort schedule by reward threshold.
    lr_schedule = sorted(lr_schedule, key=lambda x: x[0])
    
    # Current learning rate.
    current_lr = learning_rate
    
    # Initialize Adam optimizer.
    optimizer = optax.adam(learning_rate=current_lr)
    opt_state = optimizer.init(params)
    
    # Track highest reward threshold reached to prevent regression.
    highest_threshold_reached = -float('inf')
    
    # Training metrics.
    episode_rewards = []
    
    # Track maximum reward seen so far.
    max_reward = -float('inf')
    
    # Ensure video directory exists.
    video_dir = os.path.join("./lunar-lander", "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # Time tracking.
    start_time = time.time()
    
    # Training loop.
    for episode in range(num_episodes):
        # Split RNG key for this episode.
        rng_key, collect_key = jax.random.split(rng_key)
        
        # Track if this episode should be rendered.
        should_render = RENDER_ALL_EPISODES or episode % render_every == 0
        
        # Handle environment setup with proper rendering and recording
        latest_video_path = None
        
        if RECORD_BEST_EPISODE:
            # Set up recording environment
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_prefix = f"episode_{episode}_{timestamp}"
            latest_video_path = os.path.join(video_dir, f"{episode_prefix}")
            
            if should_render:
                # We want to both render and record
                # First create a human-viewable environment
                render_env = gym.make("LunarLander-v3", render_mode="human")
                # Run the episode for viewing
                gradients, total_reward, objective = collect_episode_gradients(
                    render_env, params, collect_key)
                render_env.close()
                
                # Then create a recording environment and run again to record
                record_env = gym.make("LunarLander-v3", render_mode="rgb_array")
                record_env = RecordVideo(
                    record_env,
                    video_dir,
                    name_prefix=episode_prefix,
                    episode_trigger=lambda x: True  # Always record
                )
                # We need a new key for the second run
                rng_key, record_key = jax.random.split(rng_key)
                # Just collect the episode for recording, ignoring the returns
                _, _, _ = collect_episode_gradients(record_env, params, record_key)
                record_env.close()
            else:
                # Just record without rendering
                record_env = gym.make("LunarLander-v3", render_mode="rgb_array")
                record_env = RecordVideo(
                    record_env,
                    video_dir,
                    name_prefix=episode_prefix,
                    episode_trigger=lambda x: True  # Always record
                )
                gradients, total_reward, objective = collect_episode_gradients(
                    record_env, params, collect_key)
                record_env.close()
        elif should_render:
            # Just render for viewing without recording
            render_env = gym.make("LunarLander-v3", render_mode="human")
            gradients, total_reward, objective = collect_episode_gradients(
                render_env, params, collect_key)
            render_env.close()
        else:
            # No rendering or recording
            gradients, total_reward, objective = collect_episode_gradients(
                env, params, collect_key)
        
        # Check if we've achieved a new maximum reward
        if total_reward > max_reward:
            print(f"New maximum reward: {total_reward:.2f} (previous: {max_reward:.2f})")
            max_reward = total_reward
            
            # If we recorded this episode and it's a new best, rename the video file
            if RECORD_BEST_EPISODE and latest_video_path is not None:
                # Find the recorded video file (it will have a timestamp and .mp4 extension)
                video_files = glob.glob(f"{latest_video_path}*.mp4")
                if video_files:
                    # Rename the first matching file (should only be one)
                    old_path = video_files[0]
                    new_path = os.path.join(
                        video_dir, 
                        f"max_reward_{total_reward:.2f}_{timestamp}.mp4"
                    )
                    os.rename(old_path, new_path)
                    print(f"Saved new best episode video: {new_path}")
        elif RECORD_BEST_EPISODE and latest_video_path is not None:
            # Not a new maximum reward, delete the video file
            video_files = glob.glob(f"{latest_video_path}*.mp4")
            for video_file in video_files:
                os.remove(video_file)
                print(f"Deleted non-best episode video: {video_file}")
        
        # Use Adam optimizer to update parameters (instead of direct gradient application).
        # Note: We invert the sign of gradients since Adam minimizes loss but we want to maximize rewards.
        negative_gradients = jax.tree.map(lambda g: -g, gradients)
        updates, opt_state = optimizer.update(negative_gradients, opt_state)
        params = optax.apply_updates(params, updates)
        
        episode_rewards.append(total_reward)
        
        # Check if we should update the learning rate based on average reward.
        if episode >= LR_SCHEDULE_AVERAGE_N - 1:
            avg_reward = np.mean(episode_rewards[-LR_SCHEDULE_AVERAGE_N:])
            
            # Find the appropriate learning rate based on the reward schedule.
            new_lr = current_lr
            for threshold, lr in lr_schedule:
                # Only consider thresholds we haven't reached before.
                if avg_reward >= threshold and threshold > highest_threshold_reached:
                    new_lr = lr
                    highest_threshold_reached = threshold
            
            # If learning rate changed, update the optimizer.
            if new_lr != current_lr:
                print(f"Episode {episode}: Average reward {avg_reward:.2f} reached threshold {highest_threshold_reached}. "
                    f"Changing learning rate from {current_lr} to {new_lr}.")
                current_lr = new_lr
                # Create a new optimizer with updated learning rate.
                optimizer = optax.adam(learning_rate=current_lr)
                # Reinitialize the optimizer state with the current parameters.
                opt_state = optimizer.init(params)
        
        # Save weights periodically.
        if episode % save_frequency == 0 or episode == num_episodes - 1:
            save_weights(params)
        
        # Print progress
        if episode % 1 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode > 0 else total_reward
            print(f"Episode {episode}, Loss: {objective:.4f}, Reward: {total_reward:.2f}, "
                  f"Avg Reward (last 10): {avg_reward:.2f}, LR: {current_lr}")
            
            # Print time information every 10 episodes.
            if episode % 10 == 0 and episode > 0:
                elapsed_time = time.time() - start_time
                elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                
                # Estimate remaining time.
                episodes_done = episode + 1
                episodes_remaining = num_episodes - episodes_done
                seconds_per_episode = elapsed_time / episodes_done
                remaining_time = episodes_remaining * seconds_per_episode
                remaining_str = str(timedelta(seconds=int(remaining_time)))
                
                print(f"Time elapsed: {elapsed_str}, Estimated time remaining: {remaining_str}")
    
    env.close()
    return params

def save_weights(params, base_dir="./lunar-lander"):
    """Save model weights with timestamp in filename.
    
    Args:
        params: Model parameters to save.
        base_dir: Directory to save weights in.
    
    Returns:
        Path to the saved weights file.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"lunar_lander_{timestamp}.weights.pickle"
    filepath = os.path.join(base_dir, filename)
    
    with open(filepath, "wb") as f:
        pickle.dump(params, f)
    
    print(f"Weights saved to {filepath}.")
    return filepath

def load_latest_weights(base_dir="./lunar-lander/"):
    """Load the most recent weights file from the specified directory.
    
    Args:
        base_dir: Directory to search for weight files.
    
    Returns:
        Loaded parameters or None if no weights file found.
    """
    weight_files = glob.glob(os.path.join(base_dir, "lunar_lander_*.weights.pickle"))
    
    if not weight_files:
        print("No existing weight files found.")
        return None
    
    # Sort by modification time (newest first).
    latest_file = max(weight_files, key=os.path.getmtime)
    
    print(f"Loading weights from {latest_file}.")
    with open(latest_file, "rb") as f:
        params = pickle.load(f)
    
    return params

# Run the training with a learning rate schedule.
if __name__ == "__main__":
  trained_params = train_reinforce()
  
  # Save final weights.
  save_weights(trained_params)
  
  # Evaluate the trained policy.
  eval_env = gym.make("LunarLander-v3", render_mode="human")
  rng_key = jax.random.PRNGKey(1)  # Different seed for evaluation.
  
  for _ in range(5):  # Show 5 episodes with the trained policy.
    observation, _ = eval_env.reset()
    done = False
    total_reward = 0
    
    while not done:
      rng_key, action_key = jax.random.split(rng_key)
      action = sample_action(trained_params, action_key, jnp.array(observation))
      action = int(action)
      
      observation, reward, terminated, truncated, _ = eval_env.step(action)
      total_reward += reward
      done = terminated or truncated
    
    print(f"Evaluation episode reward: {total_reward:.2f}")
  
  eval_env.close()