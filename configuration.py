import os

from env import ANTHROPHIC_API


def setup_env():
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = ANTHROPHIC_API