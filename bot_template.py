#!/usr/bin/env python3
"""Choose a strategy here; edit its decisions in the named strategy file."""

# Choices: "gold_mining", "scout_and_bases", "iron_expansion", "economy_bases".
STRATEGY = "economy_bases"

if STRATEGY == "gold_mining":
    from strategy_gold_mining import API_BASE_URL, API_KEY, calculate_actions
elif STRATEGY == "scout_and_bases":
    from strategy_scout_and_bases import API_BASE_URL, API_KEY, calculate_actions
elif STRATEGY == "iron_expansion":
    from strategy_iron_expansion import API_BASE_URL, API_KEY, calculate_actions
elif STRATEGY == "economy_bases":
    from strategy_economy_bases import API_BASE_URL, API_KEY, calculate_actions
else:
    raise ValueError('STRATEGY must be "gold_mining", "scout_and_bases", "iron_expansion", or "economy_bases".')

from bot_client import BotClient


if __name__ == "__main__":
    print(f"Running strategy: {STRATEGY}")
    try:
        BotClient(API_KEY, API_BASE_URL).connect_and_run_bot(calculate_actions)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(error)
        raise SystemExit(1)
