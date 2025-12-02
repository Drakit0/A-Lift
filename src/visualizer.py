import os
import pickle
import random
from typing import Any

import numpy as np

from warehouse_environment import WarehouseEnv
from navigation_environment import Navigation

from td_lambda import TDLambdaAgent
from qlearning import QLAgent
from sarsa import SarsaAgent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_model_path(model_name: str) -> str:
    # Add .pkl extension
    if not model_name.endswith('.pkl'):
        model_name += '.pkl'
    
    # If already absolute path and exists, return it
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '..', 'models')
    model_path = os.path.join(models_dir, os.path.basename(model_name))
    
    if os.path.exists(model_path):
        return model_path
    
    if os.path.exists(model_name):
        return model_name
    
    cwd_model_path = os.path.join('models', os.path.basename(model_name))
    if os.path.exists(cwd_model_path):
        return cwd_model_path
    
    raise FileNotFoundError(f"Model not found: {model_name}. Searched in:\n"
                           f"  - {model_path}\n"
                           f"  - {model_name}\n"
                           f"  - {cwd_model_path}")


def load_agent(model_path: str) -> Any:
    full_path = get_model_path(model_path)
    
    with open(full_path, 'rb') as f:
        agent = pickle.load(f)
    
    # Fix compatibility with older models that don't have newer attributes
    if hasattr(agent, 'feedback'):
        feedback = agent.feedback
        
        # Older models used tile coding by default
        if not hasattr(feedback, 'use_tiles'):
            feedback.use_tiles = True
            
        # Older models were for simple navigation (just_pick equivalent)
        if not hasattr(feedback, 'just_pick'):
            feedback.just_pick = True
            
        if not hasattr(feedback, 'random_objects'):
            feedback.random_objects = False
            
        if not hasattr(feedback, 'use_rich_features'):
            feedback.use_rich_features = False
        
    print(f"Loaded model: {full_path}")
    
    return agent


def get_env_type_from_model_path(model_path: str) -> str:
    filename = os.path.basename(model_path).lower()
    
    if 'qlearning' in filename or 'sarsa' in filename:
        return 'navigation'
    
    else:
        return 'warehouse'


def get_warehouse_config_from_model_path(model_path: str) -> tuple:
    filename = os.path.basename(model_path).lower()
    
    # Default to env 1
    just_pick = True
    random_objects = False
    
    # Check for env variant in filename
    if '_1_' in filename:
        just_pick = True
        random_objects = False
        
    elif '_3_' in filename:
        just_pick = False
        random_objects = True
        
    elif '_2_' in filename:
        just_pick = False
        random_objects = False
    
    return just_pick, random_objects


def visualize_agent(agent: Any, env, num_episodes: int = 5, seed: int = 42):
    
    for episode in range(num_episodes):
        try: # Seed reset
            obs, _ = env.reset(seed=seed + episode)
            
        except TypeError: # Navigation
            set_seed(seed + episode)
            obs, _ = env.reset()
        
        episode_return = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            try:
                action = agent.get_action(obs, epsilon=0.0)
                
            except TypeError:
                action = agent.get_action(obs, 0.0)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            episode_return += reward
            done = terminated or truncated
        
        status = "SUCCESS" if terminated and reward > 0 else "FAILED"
        
        print(f"Return: {episode_return:.2f} - {status}")


def list_models(models_dir: str) -> list:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_models_dir = os.path.join(script_dir, '..', models_dir)
    
    if not os.path.exists(full_models_dir):
        return []
    
    models = [f for f in os.listdir(full_models_dir) if f.endswith('.pkl')]
    
    return sorted(models)


if __name__ == "__main__":
    
    seed: int = 42
    episodes: int = 5
    
    print("\nAvailable models:")
    print("=" * 60)
    models = list_models('models')
    for i, model in enumerate(models):
        print(f"  [{i+1}] {model}")
    print("=" * 60)
    
    # Model selection
    while True:
        try:
            choice = input("\nSelect model number (or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                exit(0)
                
            idx = int(choice) - 1
            
            if 0 <= idx < len(models):
                model_path = models[idx]
                break
            
            else:
                print(f"Please enter a number between 1 and {len(models)}")
                
        except ValueError:
            print("Please enter a valid number")
    
    # Set global seeds
    set_seed(seed)
        
    # Load agent
    agent: Any = load_agent(model_path)
    
    # Determine environment type
    env_type = get_env_type_from_model_path(model_path)
    print(f"\nDetected environment type: {env_type}")
    
    # Rendering
    if env_type == 'warehouse':
        just_pick, random_objects = get_warehouse_config_from_model_path(model_path)
        print(f"Warehouse config: just_pick={just_pick}, random_objects={random_objects}")
        env = WarehouseEnv(just_pick=just_pick, random_objects=random_objects, render_mode="human")
        
    else:
        env = Navigation(render_mode="human")
    
    # Update agent's environment reference
    agent.env = env
    
    # Run visualization
    visualize_agent(agent, env, episodes, seed)
    
    env.close()
    print("\nVisualization complete.")