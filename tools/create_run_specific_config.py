# Simple utility to create a yml config file from command line arguments.

import sys
import os
import json

RUN_SPECIFIC_CONFIG_PATH="alon_configs/run_specific"

print(sys.argv[1:])
config = {}
for k, v in zip(sys.argv[1::2], sys.argv[2::2]):
    k = k.replace("--", "")
    if v.isnumeric():
        if float(v).is_integer():
            v = int(v)
        else:
            v = float(v)
    elif v.lower() == "true":
        v = True
    elif v.lower() == "false":
        v = False
    config[k] = v

print(config)

assert("save" in config)
save_path = os.path.join(RUN_SPECIFIC_CONFIG_PATH, config["save"].split("/")[-1] + ".yml")
with open(save_path, "w") as f:
    json.dump(config, f, indent=2)

