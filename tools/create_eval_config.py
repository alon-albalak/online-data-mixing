# Simple utility that takes an existing yml config file and adds evaluation specific parameters.

import argparse
import json
import os

def get_step_from_path(path):
    top_dir = path.split("/")[-1]
    if "global_step" in top_dir:
        return int(top_dir.split("global_step")[-1])
    else:
        return get_step_from_path("/".join(path.split("/")[:-1])) 

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the config file to modify.")

args = parser.parse_args()
model_step = get_step_from_path(args.config_path)

with open(args.config_path, "r") as f:
    config = json.load(f)

if "seed" not in config.keys():
    config["seed"] = 1234

config["wandb_run_name"] = f"seed{config['seed']}_eval"

eval_results_prefix = os.path.join(config["save"], f"step{model_step}")
config['load'] = config['save']
config['eval_results_prefix'] = eval_results_prefix

save_path = args.config_path.replace(".yml", "_eval.yml")
with open(save_path, "w") as f:
    json.dump(config, f, indent=2)