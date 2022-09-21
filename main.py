"""MVP for trying out training on a gpu"""

import os
import json
from torch import cuda

def main(config_path):

    with open(config_path, "r") as fh:
        config = json.load(fh)

    print(config)
    print("GPU available:", cuda.is_available())


if __name__ == "__main__":
    main("./conf/config.json")