import json
import collections
import argparse
import random
import numpy as np
import requests
import re
import os

def your_netid():
    YOUR_NET_ID = 'SD6701'
    return YOUR_NET_ID

def your_hf_token():
    return os.environ.get("HF_TOKEN", "")


# for adding small numbers (1-6 digits) and large numbers (7 digits), wcna rite prompt prefix and prompt suffix separately.
def your_prompt():
    here = os.path.dirname(__file__)
    prompt_path = os.path.join(here, "prompt.txt")
    with open(prompt_path, "r") as f:
        prefix = f.read()
    suffix = "?\nAnswer: "
    return prefix, suffix


def your_config():
    """Returns a config for prompting api
    Returns:
        For both short/medium, long: a dictionary with fixed string keys.
    Note:
        do not add additional keys. 
        The autograder will check whether additional keys are present.
        Adding additional keys will result in error.
    """
    config = {
        'max_tokens': 50, # max_tokens must be >= 50 because we don't always have prior on output length 
        'temperature': 0.1,
        'top_k': 50,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
        'stop': []}
    
    return config


def your_pre_processing(s):
    return s

    
def your_post_processing(output_string):
    """Returns the post processing function to extract the answer for addition
    Returns:
        For: the function returns extracted result
    Note:
        do not attempt to "hack" the post processing function
        by extracting the two given numbers and adding them.
        the autograder will check whether the post processing function contains arithmetic additiona and the graders might also manually check.
    """
    stripped = output_string.strip()
    if not stripped:
        return 0

    for line in stripped.splitlines():
        match = re.search(r"answer:\s*(\d{8})", line.strip(), flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    match = re.search(r"^\s*(\d{8})\s*$", stripped.splitlines()[0])
    if match:
        return int(match.group(1))

    match = re.search(r"\d{8}", stripped)
    if match:
        return int(match.group(0))

    match = re.search(r"\d{7}", stripped)
    if match:
        return int(match.group(0))

    match = re.search(r"\d+", stripped)
    if match:
        return int(match.group(0))

    return 0
