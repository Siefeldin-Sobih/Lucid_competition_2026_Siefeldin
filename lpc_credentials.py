"""Read the team key without putting it in either strategy's source code."""
import os
from pathlib import Path


def load_api_key():
    # An explicit terminal setting can override the saved team key.
    key = os.environ.get("API_KEY", "").strip()
    if key and key != "replace-with-api-key":
        return key

    # Resolve beside THIS file, so launching from another folder still works.
    key_file = Path(__file__).with_name(".lpc_api_key")
    try:
        key = key_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        key = ""

    if not key:
        raise ValueError("No team key saved. Set API_KEY or add it to .lpc_api_key beside the bot files.")
    return key
