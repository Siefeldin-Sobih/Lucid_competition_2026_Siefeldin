#!/usr/bin/env python3
"""Simple economy: three workers, useful depots, four-worker cap, bank gold.

Run directly, or select "economy_bases" in bot_template.py. All tuning values
are below. No network connection is made until the __main__ block runs.
"""
import os

from bot_client import BotClient, mars_view
from lpc_credentials import load_api_key
# Reuse the original's route and price helpers; its strategy is not called.
from strategy_gold_mining import _DIRECTIONS, _search, _spawn_price

API_KEY = load_api_key()
API_BASE_URL = os.environ.get("LUCID_API_BASE_URL", "https://lpc.sh")

# These are experimental strategy choices, not competition rules.
OPENING_ROVERS = 3
TARGET_ROVERS = 4                  # No fifth-rover funding or special-case logic.
EXPANSION_MIN_REMAINING = 180
BASE_MIN_REMAINING = 90            # Below this, build only with funded, timed payback.
BASE_IRON_RESERVE = 6              # Refill only one base's cost, not a fifth rover.
NEAR_GOLD_DISTANCE = 5              # Includes mountain/crater travel delays.
SCOUT_DETOUR_TURNS = 3              # Then commit to known distant gold.
SCOUT_STOP_REMAINING = 100
FRESH_ORE_TURNS = 12
MAX_BASES = 9
BASE_MIN_GOLD = 16                 # At least two loads must benefit.
BASE_MIN_ROUND_TRIP_SAVING = 4
BASE_MIN_NET_SAVING = 8
BASE_IRON_WORK = 6                 # Charge at least six mining turns per new base.
BASE_MAX_APPROACH = 10
BASE_RADIUS = 4
RETURN_BUFFER = 5
NEAR_GOLD_BONUS = 1.1              # Prefer close gold, but do not chase scraps.
DELIVERY_BASE_MIN_GOLD = 4
DELIVERY_BASE_MIN_SAVING = 4

_memory = {}


def _position(entity):
    return entity["position"]["x"], entity["position"]["y"]


def _load(worker):
    return sum(worker.get("inventory", {}).values())


def _bankable_gold(stock, first_trip, outward, inward, capacity, turns):
    """Estimate one rover's deliveries; count only gold banked before time runs out."""
    stock = dict(stock)
    total = 0
    while turns > 0:
        choices = []
        for q, available in stock.items():
            travel = first_trip.get(q, 1000) + inward.get(q, 1000) + 1
            amount = min(available, capacity, turns-travel)
            if amount > 0:
                choices.append((amount/(travel+amount), q, amount, travel+amount))
        if not choices:
            break
        _, q, amount, duration = max(choices)
        total += amount
        stock[q] -= amount
        turns -= duration
        first_trip = outward
    return total


def _choose_base(view, workers, visible, cells, bases, enemy, gold,
                 routes, home_distance, base_outward, neighbors, costs, remaining):
    """Shortlist sites cheaply; evaluate four with actual round-trip costs."""
    occupied = {_position(w): w["unitId"] for w in view["units"]}
    builders = [w for w in workers if w["unitId"] in routes
                and not w.get("inventory", {}).get("iron", 0)]
    team = workers[0]["teamId"] if workers else None
    shortlist = []
    for site in visible:
        if cells[site]["type"] != "empty" or site in bases or site not in home_distance:
            continue
        if any(abs(q[0]-site[0]) + abs(q[1]-site[1]) <= 2 for q in enemy):
            continue
        candidates = [w for w in builders if routes[w["unitId"]][0].get(site, 1000) <= BASE_MAX_APPROACH
                      and (site not in occupied or (occupied[site] == w["unitId"]
                           and sum(_position(v) == site for v in view["units"]) == 1))
                      # A loaded returning rover may build if doing so gets it
                      # to a deposit point sooner than travelling to the old base.
                      and (not (w["unitId"] in _memory["returns"] or _load(w) >= w.get("inventoryCapacity", 8))
                           or routes[w["unitId"]][0][site] + 1 < home_distance.get(_position(w), 0))]
        if not candidates:
            continue
        builder = min(candidates, key=lambda w: (routes[w["unitId"]][0][site], w["unitId"]))
        # A new base can overlap an enemy claim, making that ore mineable.
        # Do not count enemy-only ore beyond the new base's claim radius.
        near = {}
        for q, c in gold.items():
            if max(abs(q[0]-site[0]), abs(q[1]-site[1])) > BASE_RADIUS:
                continue
            owners = {b["teamId"] for p, b in bases.items() if abs(p[0]-q[0])+abs(p[1]-q[1]) <= 2}
            if not owners or team in owners or abs(q[0]-site[0])+abs(q[1]-site[1]) <= 2:
                near[q] = c
        if sum(c["amount"] for c in near.values()) < BASE_MIN_GOLD:
            continue
        approach = routes[builder["unitId"]][0][site]
        rough = sum(c["amount"]/8 * max(0, 2*(home_distance[q]
                    - max(abs(q[0]-site[0]), abs(q[1]-site[1])))) for q, c in near.items())
        shortlist.append((rough - approach - 1, site, builder["unitId"], near))
    shortlist.sort(key=lambda x: (-x[0], x[1], x[2]))
    # Retain a viable previous job to prevent chasing changing destinations.
    previous = _memory.get("base_job")
    old = [x for x in shortlist if previous and x[1] == previous["site"] and x[2] == previous["builder"]]
    options = old[:1] + [x for x in shortlist if x not in old][:4-len(old[:1])]
    best = None
    for _, site, builder, near in options:
        outward, _ = _search([site], neighbors, costs)
        inward, _ = _search([site], neighbors, costs, reverse=True)
        savings = {q: base_outward[q] + home_distance[q] - outward[q] - inward[q]
                   for q in near if q in outward and q in base_outward}
        useful = {q for q, saving in savings.items() if saving >= BASE_MIN_ROUND_TRIP_SAVING}
        useful_gold = sum(near[q]["amount"] for q in useful)
        # Payback must cover travel, construction, and at least the six actions
        # needed to mine its iron. Actual iron-gathering travel is not modeled.
        net = (sum(near[q]["amount"]/8 * savings[q] for q in useful)
               - routes[builder][0][site] - 1 - BASE_IRON_WORK)
        # The chosen builder can reach the site and these deposits. A single
        # productive miner is enough if the deliveries repay the full setup.
        if useful_gold < BASE_MIN_GOLD or net < BASE_MIN_NET_SAVING:
            continue
        job = {"site": site, "builder": builder, "net_saving": net, "gold": useful_gold}
        if remaining <= BASE_MIN_REMAINING:
            # Late bases must increase banked gold, not merely look useful on a map.
            # Compare the same rover and ore with construction versus existing bases.
            w = next(w for w in workers if w["unitId"] == builder)
            capacity = w.get("inventoryCapacity", 8)
            carried = w.get("inventory", {}).get("gold", 0)
            available = remaining - RETURN_BUFFER - max(0, w.get("readyOnTurn", 0)-view["turn"])
            stock = {q: near[q]["amount"] for q in useful}
            build_time = routes[builder][0][site] + 1 + bool(carried)
            new_gold = _bankable_gold(stock, outward, outward, inward, capacity, available-build_time)
            old_time = home_distance[_position(w)] + 1 if carried else 0
            first_trip = base_outward if carried else routes[builder][0]
            old_gold = _bankable_gold(stock, first_trip, base_outward, home_distance,
                                      capacity, available-old_time)
            if new_gold < capacity or new_gold <= old_gold:
                continue
            job["deadline_checked"] = True
            job["expected_extra_gold"] = new_gold-old_gold
        if old and site == old[0][1] and builder == old[0][2]:
            return job
        if best is None or net > best["net_saving"]:
            best = job
    return best


def _delivery_base(view, workers, visible, cells, bases, enemy, routes, home_distance):
    """Fallback: spend already-banked iron to shorten a loaded rover's return.

    This needs no guess about future ore. It checks only the current delivery,
    a nearby legal building tile, and enough time to build and deposit.
    """
    best = None
    for w in workers:
        uid, p = w["unitId"], _position(w)
        inv = w.get("inventory", {})
        if uid not in routes or inv.get("gold", 0) < DELIVERY_BASE_MIN_GOLD or inv.get("iron", 0):
            continue
        # Do not interrupt productive mining with a partial load.
        cell = cells.get(p, {})
        if (_load(w) < w.get("inventoryCapacity", 8) and uid not in _memory["returns"]
                and cell.get("type") == "gold" and cell.get("amount", 0) > 0):
            continue
        for site in [p] + [(p[0]+dx, p[1]+dy) for dx, dy in _DIRECTIONS]:
            if site not in visible or cells[site]["type"] != "empty" or site in bases:
                continue
            if any(_position(v) == site and v["unitId"] != uid for v in view["units"]):
                continue
            if any(abs(q[0]-site[0])+abs(q[1]-site[1]) <= 2 for q in enemy):
                continue
            approach = routes[uid][0].get(site, 1000)
            saving = home_distance.get(p, 0) - approach - 1
            if saving >= DELIVERY_BASE_MIN_SAVING and (best is None or saving > best["net_saving"]):
                best = {"site": site, "builder": uid, "net_saving": saving,
                        "gold": inv["gold"], "delivery_only": True}
    return best


def _exploration_target(p, distances, cells, seen, visits, home_distance,
                        width, height, remaining, assigned, turn, far_goal=None, budget=None):
    """Explore separately. A detour toward known gold has a strict time budget."""
    best, best_score = None, -1
    for q, distance in distances.items():
        if q == p or distance + home_distance.get(q, 1000) + RETURN_BUFFER >= remaining:
            continue
        if budget is not None and distance > budget:
            continue
        if far_goal and ((q[0]-p[0])*(far_goal[0]-p[0]) + (q[1]-p[1])*(far_goal[1]-p[1]) < 0):
            continue
        gain = sum(0 <= q[0]+dx < width and 0 <= q[1]+dy < height
                   and (q[0]+dx, q[1]+dy) not in cells
                   for dx in range(-3, 4) for dy in range(-3, 4) if abs(dx)+abs(dy) <= 3)
        # Revisit stale territory for respawns when there is no fresh frontier.
        age = max(0, turn - seen.get(q, turn))
        if budget is not None and not gain:
            continue
        separation = min((max(abs(q[0]-v[0]), abs(q[1]-v[1])) for v in assigned), default=6)
        score = (gain*4 + min(age, 80)/10) * (0.4 + min(separation, 6)/6)
        score /= (distance + 3) * (1 + visits.get(q, 0)*0.2)
        if score > best_score:
            best, best_score = q, score
    return best


def calculate_actions(game_state, identity):
    view = mars_view(game_state)
    if view is None:
        return {}
    team, turn = identity["teamId"], view["turn"]
    grid, match = view["map"], game_state["match"]
    width, height = grid["width"], grid["height"]
    key = (match["matchId"], team, width, height)
    if _memory.get("key") != key or turn < _memory.get("turn", -1):
        _memory.clear()
        _memory.update(key=key, cells={}, seen={}, bases={}, visits={}, returns=set(),
                       iron_worker=None, base_job=None, detours={}, decisions={})
    _memory["turn"], _memory["decisions"] = turn, {}
    remaining = max(0, match.get("maxTurns", 480) - turn)
    visible = set()
    for index, cell in enumerate(grid["cells"]):
        if cell["type"] != "hidden":
            p = index % width, index // width
            visible.add(p)
            _memory["cells"][p], _memory["seen"][p] = dict(cell), turn
    cells, seen = _memory["cells"], _memory["seen"]
    current_bases = {_position(b): b for b in view["bases"]}
    for p in list(_memory["bases"]):
        if p in visible and p not in current_bases:
            del _memory["bases"][p]
    _memory["bases"].update(current_bases)
    bases = _memory["bases"]
    friendly = {p for p, b in bases.items() if b["teamId"] == team}
    enemy = {p for p, b in bases.items() if b["teamId"] != team}
    workers = sorted((w for w in view["units"] if w["teamId"] == team), key=lambda w: w["unitId"])
    actions = {w["unitId"]: {"type": "pass"} for w in workers}
    ids = set(actions)
    _memory["returns"].intersection_update(ids)
    _memory["detours"] = {uid: d for uid, d in _memory["detours"].items() if uid in ids}
    for w in workers:
        p = _position(w)
        _memory["visits"][p] = _memory["visits"].get(p, 0) + 1
    costs = {p: 3 if c["type"] in ("mountain", "crater") else 1 for p, c in cells.items()
             if c["type"] not in ("hidden", "void") and p not in enemy}
    neighbors = {p: [(p[0]+dx, p[1]+dy) for dx, dy in _DIRECTIONS if (p[0]+dx, p[1]+dy) in costs]
                 for p in costs}
    starts = sorted(friendly & costs.keys())
    if not starts:
        return actions
    home_distance, _ = _search(starts, neighbors, costs, reverse=True)
    base_outward, _ = _search(starts, neighbors, costs)
    routes = {w["unitId"]: _search([_position(w)], neighbors, costs)
              for w in workers if _position(w) in costs}

    def eligible(p):
        claimants = {b["teamId"] for q, b in bases.items() if abs(q[0]-p[0])+abs(q[1]-p[1]) <= 2}
        return not claimants or team in claimants

    ores = {p: c for p, c in cells.items() if c["type"] in ("gold", "iron")
            and c.get("amount", 0) > 0 and p in home_distance and eligible(p)}
    gold = {p: c for p, c in ores.items() if c["type"] == "gold"}
    # Construction may open currently enemy-claimed ore; ordinary miners still
    # use the stricter eligible gold list until the new base actually exists.
    expansion_gold = {p: c for p, c in cells.items() if c["type"] == "gold"
                      and c.get("amount", 0) > 0 and p in home_distance
                      and turn-seen[p] <= FRESH_ORE_TURNS}
    bank = view.get("teamResources", {}).get(team, {})
    bank_iron = bank.get("iron", 0)
    price = _spawn_price(max(0, len(workers)-1))
    invest = remaining > EXPANSION_MIN_REMAINING
    # Simple build order: get three immediately. Allow ONE useful forward base
    # before rover four. Once it exists, recruit four before further bases.
    job = None
    # Late relocation uses existing iron; do not start an iron expedition for it.
    base_invest = len(friendly) < MAX_BASES and (remaining > BASE_MIN_REMAINING or bank_iron >= 6)
    early_base = len(workers) == OPENING_ROVERS and len(friendly) == 1
    if (len(workers) >= TARGET_ROVERS or early_base) and base_invest:
        job = _choose_base(view, workers, visible, cells, bases, enemy, expansion_gold,
                           routes, home_distance, base_outward, neighbors, costs, remaining)
    # A late loaded rover can use iron already in the bank even when normal
    # expansion has stopped. Construction must finish soon enough to deposit.
    if job is None and len(workers) >= TARGET_ROVERS and len(friendly) < MAX_BASES and bank_iron >= 6:
        candidate = _delivery_base(view, workers, visible, cells, bases, enemy, routes, home_distance)
        if candidate and routes[candidate["builder"]][0][candidate["site"]] + 3 <= remaining:
            job = candidate
    _memory["base_job"] = job
    want_spawn = invest and len(workers) < TARGET_ROVERS and job is None
    funding_goal = (BASE_IRON_RESERVE if job else price if want_spawn else
                    BASE_IRON_RESERVE if len(workers) >= TARGET_ROVERS and base_invest else 0)
    carried_iron = sum(w.get("inventory", {}).get("iron", 0) for w in workers)
    iron_to_mine = max(0, funding_goal-bank_iron-carried_iron)

    # Keep a supplier carrying iron on its delivery. Otherwise select the rover
    # with the best mining-and-return trip, not the first rover in ID order.
    iron_id, iron_target, iron_score = None, None, -1
    previous_iron = _memory["iron_worker"]
    carrying = [w for w in workers if w.get("inventory", {}).get("iron", 0) > 0
                and not w.get("inventory", {}).get("gold", 0)]
    if carrying:
        supplier = min(carrying, key=lambda w: (w["unitId"] != previous_iron, w["unitId"]))
        iron_id = supplier["unitId"]
    locked_supplier = iron_id
    for w in workers:
        uid = w["unitId"]
        if uid not in routes or w.get("inventory", {}).get("gold", 0) or (locked_supplier and uid != locked_supplier):
            continue
        if uid in _memory["returns"] or _load(w) >= w.get("inventoryCapacity", 8):
            continue
        for q, c in ores.items():
            if c["type"] != "iron" or q not in routes[uid][0] or iron_to_mine <= 0:
                continue
            amount = min(c["amount"], iron_to_mine, w.get("inventoryCapacity", 8)-_load(w))
            cost = routes[uid][0][q] + amount + home_distance[q] + 1 + max(0, w.get("readyOnTurn", 0)-turn)
            score = amount/cost
            if uid == previous_iron:
                score *= 1.1
            if cost + RETURN_BUFFER < remaining and score > iron_score:
                iron_id, iron_target, iron_score = uid, q, score
    _memory["iron_worker"] = iron_id
    _memory["investment"] = {"want_spawn": want_spawn, "funding_goal": funding_goal,
                               "iron_to_mine": iron_to_mine}

    # An empty rover recruits where it stands. Prefer one not actively mining.
    candidates = [w for w in workers if not _load(w) and w["unitId"] in routes
                  and turn >= w.get("readyOnTurn", 0)
                  and len(workers) < TARGET_ROVERS]
    spawn_id = min(candidates, key=lambda w: (_position(w) in ores, w["unitId"]))["unitId"] if candidates else None
    reserved, scout_targets, constructed = {}, [], False
    building_now = bank_iron >= 6 and job and next((_position(w) == job["site"] for w in workers if w["unitId"] == job["builder"]), False)
    # Supplier first, then rovers nearest gold, so distant rovers do not claim a
    # nearby rover's deposit before that rover gets its decision.
    workers.sort(key=lambda w: (w["unitId"] != iron_id,
                  min((routes.get(w["unitId"], ({}, {}))[0].get(q, 1000) for q in gold), default=1000), w["unitId"]))
    for w in workers:
        uid, p, inv = w["unitId"], _position(w), w.get("inventory", {})
        load, capacity = _load(w), w.get("inventoryCapacity", 8)
        if uid not in routes or turn < w.get("readyOnTurn", 0):
            continue
        distances, first = routes[uid]
        homes = friendly & distances.keys()
        home = min(homes, key=distances.get) if homes else None
        return_cost = distances.get(home, 1000)

        def go(target, reason):
            step = first.get(target)
            if step and not (building_now and uid != job["builder"] and step == job["site"]):
                actions[uid] = {"type": "move", "direction": _DIRECTIONS[(step[0]-p[0], step[1]-p[1])]}
            _memory["decisions"][uid] = reason

        if load and p in friendly:
            actions[uid] = {"type": "deposit"}
            _memory["returns"].discard(uid)
            _memory["detours"].pop(uid, None)
            _memory["decisions"][uid] = "deposit"
            continue
        # A funded base job can shorten a loaded rover's delivery route. Handle
        # it before the ordinary full-inventory return decision, subject to the
        # endgame buffer. The site was already checked for legal construction.
        deadline_ok = (job and distances.get(job["site"], 1000) + 3 <= remaining
                       if job and (job.get("delivery_only") or job.get("deadline_checked"))
                       else remaining > return_cost + RETURN_BUFFER)
        if (job and uid == job["builder"] and bank_iron >= 6 and not constructed and deadline_ok):
            _memory["returns"].discard(uid)
            if p == job["site"]:
                actions[uid] = {"type": "build-base"}
                bank_iron -= 6
                constructed = True
                _memory["decisions"][uid] = "build useful forward base"
            else:
                go(job["site"], "travel to forward base")
            continue
        if load >= capacity or (load and remaining <= return_cost + RETURN_BUFFER):
            _memory["returns"].add(uid)
        # Deliver investment iron as soon as this load meets the bank shortfall.
        if inv.get("iron", 0) and (iron_to_mine == 0 or iron_target is None):
            _memory["returns"].add(uid)
        if not load:
            _memory["returns"].discard(uid)
        if uid in _memory["returns"]:
            if home is not None:
                go(home, "deliver load")
            continue
        if remaining <= return_cost + RETURN_BUFFER:
            continue
        if want_spawn and uid == spawn_id and bank_iron >= price and not constructed:
            actions[uid] = {"type": "spawn"}
            bank_iron -= price
            constructed = True
            _memory["decisions"][uid] = "recruit toward four"
            continue
        if uid == iron_id and iron_target is not None and iron_to_mine > 0:
            if p == iron_target:
                actions[uid] = {"type": "mine"}
                _memory["decisions"][uid] = "mine expansion iron"
            else:
                go(iron_target, "travel to expansion iron")
            continue

        # Each rover evaluates unassigned gold independently. Mining a known
        # eligible deposit underfoot takes priority over moving to another one.
        choices = []
        for q, c in gold.items():
            if q not in distances:
                continue
            amount = min(capacity-load, c["amount"]-reserved.get(q, 0),
                         remaining-distances[q]-home_distance[q]-RETURN_BUFFER)
            if amount <= 0:
                continue
            score = (inv.get("gold", 0)+amount)/(distances[q]+amount+home_distance[q]+1)
            score /= 1 + max(0, turn-seen[q]-FRESH_ORE_TURNS)/80
            if distances[q] <= NEAR_GOLD_DISTANCE:
                score *= NEAR_GOLD_BONUS
            choices.append((score, q, amount))
        underfoot = [c for c in choices if c[1] == p and p in visible]
        # Mine underfoot first. Otherwise compare ALL trips by value; being
        # within five turns is a small bonus, not permission to ignore better ore.
        candidates = underfoot or choices
        best = max(candidates, default=None, key=lambda c: (c[0], c[1]))
        if best and load and best[1] != p and inv.get("gold", 0)/(return_cost+1) >= best[0]:
            _memory["returns"].add(uid)
            go(home, "bank partial load before a costly detour")
            continue
        if best and not load and distances[best[1]] > NEAR_GOLD_DISTANCE and remaining > SCOUT_STOP_REMAINING:
            detour = _memory["detours"].get(uid)
            if not detour or detour["goal"] not in gold or gold[detour["goal"]]["amount"] <= reserved.get(detour["goal"], 0):
                detour = {"start": turn, "goal": best[1]}
                _memory["detours"][uid] = detour
            # Once the bounded search ends, commit to that known deposit.
            committed = next((c for c in choices if c[1] == detour["goal"]), None)
            budget = SCOUT_DETOUR_TURNS - (turn-detour["start"])
            target = _exploration_target(p, distances, cells, seen, _memory["visits"], home_distance,
                                         width, height, remaining, scout_targets, turn, detour["goal"], budget) if budget > 0 else None
            if target is not None:
                scout_targets.append(target)
                go(target, "short scout detour toward known gold")
                continue
            if committed:
                best = committed
        if best:
            _, q, amount = best
            reserved[q] = reserved.get(q, 0) + amount
            if q == p:
                actions[uid] = {"type": "mine"}
                _memory["detours"].pop(uid, None)
                _memory["decisions"][uid] = "mine gold"
            else:
                go(q, "travel to assigned gold")
            continue
        if load and home is not None:
            _memory["returns"].add(uid)
            go(home, "bank partial load")
            continue
        target = _exploration_target(p, distances, cells, seen, _memory["visits"], home_distance,
                                     width, height, remaining, scout_targets, turn)
        if target is not None:
            scout_targets.append(target)
            go(target, "scout independently")
    return actions


if __name__ == "__main__":
    print("Economy and bases: three early, useful depot, four-worker cap, mine and bank gold.")
    try:
        BotClient(API_KEY, API_BASE_URL).connect_and_run_bot(calculate_actions)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(error)
        raise SystemExit(1)
