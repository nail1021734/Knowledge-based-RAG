import json
from typing import Dict, List


def save_json(data: List[dict], file_path: str):
    """
    Save a list of dictionaries to a JSON file.
    Args:
        data (List[dict]): The data to save.
        file_path (str): The path to the JSON file.
    """
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(file_path: str) -> List[dict]:
    """
    Load a list of dictionaries from a JSON file.
    Args:
        file_path (str): The path to the JSON file.
    Returns:
        List[dict]: The loaded data.
    """
    with open(file_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data