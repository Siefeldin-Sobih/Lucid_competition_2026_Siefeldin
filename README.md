# Random Bagles — Lucid Competition Bot

A small collection of rule-based Python bots built for Lucid's real-time strategy competition. The final bot's decision rules helped us advance to the semifinals.

The project explores a simple question: when should a rover mine, explore, return its cargo, or invest in another rover or base? The strategies balance resource collection, travel time, and the time left in a match.

## What's here

- `bot_template.py` — entry point; currently selects `economy_bases`.
- `strategy_economy_bases.py` — economy and base-expansion strategy.
- `strategy_economy_five_rovers.py` — experimental variant with a conditional fifth rover.
- `strategy_gold_mining.py`, `strategy_iron_expansion.py`, and `strategy_scout_and_bases.py` — alternative strategies and shared helpers.
- `bot_client.py` — competition client.
- `lpc_credentials.py` — reads credentials from your environment or a local file.

These are the local development versions. The selected default is preserved; this archive does not identify which exact revision was submitted in the semifinals.

## Run

Use Python 3.10 or newer. The code uses Python's standard library.

1. Clone this repository and open a terminal in its directory.
2. Set `API_KEY` to your own competition key, or save it in a local `.lpc_api_key` file.
3. Run `python3 bot_template.py`.

To try the fifth-rover variant, run `python3 strategy_economy_five_rovers.py`. Other strategies can be selected in `bot_template.py`.

The default server is `https://lpc.sh`; override it with `LUCID_API_BASE_URL` if needed. Running a bot requires a valid key and an available competition server.

Credentials, caches, and editor backups are excluded from this repository.

