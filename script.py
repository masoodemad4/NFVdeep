import inspect
import logging
import argparse
from pathlib import Path
import sys
import os
import stable_baselines3
from stable_baselines3.common.callbacks import EvalCallback

from nfvdeep.agent import baselines
from nfvdeep.agent.baselines import BaselineHeuristic
from nfvdeep.environment.env import Env
from nfvdeep.environment.arrival import *
from nfvdeep.environment.monitor import EvalLogCallback, StatsWrapper
from stable_baselines3.common.env_checker import check_env


parser = argparse.ArgumentParser()

parser.add_argument("--total_train_timesteps", type=int, default=10000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--overlay", type=str)
parser.add_argument("--requests", type=str)
parser.add_argument("--agent", type=str)
parser.add_argument("--n_eval_episodes", type=int, default=5)
parser.add_argument("--eval_freq", type=int, default=10000)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--debug", action="store_false")
parser.add_argument("--logs",type=str, default="/workspaces/NFVdeep")

args = parser.parse_args()


def main():
    logging.basicConfig()
    debug_level = logging.INFO if args.debug else logging.DEBUG
    logging.getLogger().setLevel(debug_level)
    if args.logs: 
        print ("logging output to  " + args.logs + "/output.txt")
        sys.stdout = open((args.logs+'/output.txt'),'wt') 
    else: sys.stdout = sys.__stdout__
    Path(f"{args.output}/logs").mkdir(exist_ok=True, parents=True)
    Path(f"{args.output}/evaluation").mkdir(exist_ok=True, parents=True)

    with open(Path(args.requests), "r") as file:
        arrival_config = json.load(file)

        
    arrival_config["seed"] = args.seed
    #emad print('this is arg',args.seed)
    env = Env(args.overlay, arrival_config)
    #check_env(env)
    #print('env is for seed',arrival_config ,env)
    arrival_config["seed"] = args.seed + 1
    #print('env is for seed',arrival_config ,env)
    eval_env = StatsWrapper(Env(args.overlay, arrival_config))
    
    eval_log_callback = EvalLogCallback(log_path=f"{args.output}/evaluation")
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        log_path=f"{args.output}/evaluation",
        eval_freq=args.eval_freq,
        deterministic=False,
        render=False,
        callback_after_eval=eval_log_callback,
    )

    # if args.agent in [name for name, _ in inspect.getmembers(baselines)]:
    #     policy = getattr(baselines, args.agent)
    #     agent = BaselineHeuristic(
    #         **{
    #             "policy": policy,
    #             "env": env,
    #             "verbose": 1,
    #             "tensorboard_log": f"{args.output}/logs",
    #         }
    #     )

    if args.agent in [name for name, _ in inspect.getmembers(stable_baselines3)]:
        Agent = getattr(stable_baselines3, args.agent)
        agent = Agent(
            **{
                "policy": "MlpPolicy",
                "env": env,
                "verbose": 1,
                "tensorboard_log": f"{args.output}/logs",
            }
        )

    else:
        raise ValueError('agent is not found')

    agent.learn(
        total_timesteps=args.total_train_timesteps,
        tb_log_name=args.agent,
        callback=eval_callback,
    )



if __name__ == "__main__":
    main()
    if args.logs: 
        sys.stdout = sys.__stdout__
        print('Simulation has ran completely and output file created to:',args.logs + "/output.txt")
        
         
        
