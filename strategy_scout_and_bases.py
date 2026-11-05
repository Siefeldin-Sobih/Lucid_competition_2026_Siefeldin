#!/usr/bin/env python3
from __future__ import annotations

import os
from heapq import heappop, heappush
from collections.abc import Sequence
from typing import Optional
from lpc_credentials import load_api_key
from bot_client import (
    ActionsByEntity,
    BaseLocation,
    BotClient,
    BotState,
    Coordinate,
    CoordinateGrid,
    GameMap,
    Identity,
    MarsView,
    MovementState,
    OreState,
    Resource,
    TerritoryState,
    mars_view,
)

# Both strategies use the same saved key; API_KEY in the terminal can override it.
API_KEY = load_api_key()
API_BASE_URL = os.environ.get("LUCID_API_BASE_URL", "https://lpc.sh")

# Demo strategy controls. These are hypotheses to test, not official game rules.
# Movement distances below count terrain delays, not just straight-line tiles.
SCOUT_GOLD_MAX_TRAVEL = 5
SCOUT_MIN_GOLD = 1              # Any unassigned eligible gold can trigger a detour.
SCOUT_STOP_REMAINING = 100      # Let the scout focus on collecting near the end.
INVEST_STOP_REMAINING = 180
FOURTH_MIN_REMAINING = 240
FOURTH_BANKED_GOLD = 24
FOURTH_MIN_AVAILABLE_GOLD = 16  # Two unassigned full loads must be within reach.
FOURTH_GOLD_MAX_TRAVEL = 8
IRON_MAX_TRAVEL = 5
IRON_MAX_RETURN = 6
BASE_MAX_APPROACH = 12
BASE_GOLD_RADIUS = 4            # Diagonal tile distance, used to shortlist sites.
BASE_MIN_GOLD = 12
BASE_MIN_HOME_DISTANCE = 6
BASE_MIN_NET_SAVING = 8         # Approximate travel turns saved after setup cost.
MAX_BASES = 3
ORE_FRESH_TURNS = 12

# Strategy memory lives only in this process and resets when the match changes.
_memory: dict = {}
_DIRECTIONS = {
    (-1, -1): "north-west", (0, -1): "north", (1, -1): "north-east",
    (-1, 0): "west", (1, 0): "east",
    (-1, 1): "south-west", (0, 1): "south", (1, 1): "south-east",
}


def _search(start, neighbors, costs, reverse=False):
    """One shortest-path search for every candidate, rather than one per ore."""
    # CHUNK 3: Dijkstra's shortest-path search.
    # start: a list of (x, y) starting tiles; one rover or several bases.
    # neighbors[p]: known tiles directly reachable from tile p in one move.
    # costs[p]: turns charged for entering p, including any terrain cooldown.
    # reverse: search backward from bases to measure travel TOWARD those bases.

    # Each starting tile is zero turns away from itself.
    distances = {p: 0 for p in start}
    # first[destination] will store the FIRST step of its cheapest known route.
    first = {}
    # A heap is a queue that gives us the smallest (travel_time, tile) first.
    # heappush adds work; heappop takes the cheapest unfinished work.
    queue = [(0, p) for p in start]
    # All starting distances are zero; sorted tuples form a valid heap.
    queue.sort()
    while queue:
        distance, p = heappop(queue)
        # A better route may have been discovered since this entry was queued.
        # Skip the old entry rather than expanding an inferior route again.
        if distance != distances[p]:
            continue
        for q in neighbors[p]:
            # Forward travel p -> q pays q's entry cost. In reverse search,
            # this edge represents real travel q -> p, so pay p's entry cost.
            candidate = distance + costs[p if reverse else q]
            # An unseen tile has an initial distance of infinity. Update only
            # when this candidate improves the cheapest route recorded so far.
            if candidate < distances.get(q, float("inf")):
                distances[q] = candidate
                # Direct neighbor of a root: first step is q. For a longer
                # route, inherit the first step of the route that reached p.
                first[q] = q if distance == 0 else first[p]
                heappush(queue, (candidate, q))
    # In reverse searches we use distances only; first is not a homeward route.
    return distances, first


def _spawn_price(recruited):
    # CHUNK 4: Price of the NEXT rover, given the number already recruited.
    # Zero recruited -> 5; one -> 6; two -> 11; three -> 17.
    a, b = 5, 6
    for _ in range(recruited):
        # Python evaluates both right-hand values before replacing a and b.
        a, b = b, a + b
    return a


def _base_plan(scout, visible, cells, bases, enemy, ores,
               home_distance, routes, neighbors, costs, seen, turn, occupied, previous_site):
    """Choose a legal forward site using a small number of exact route searches.

    First rank possible sites cheaply. Then evaluate the best six using actual
    terrain costs. This bounds the work we do during a one-second turn.
    Returns {site, gold, net_saving}, or None if no worthwhile site is known.
    """
    distances = routes[scout][0]
    gold = {q: c for q, c in ores.items() if c["type"] == "gold"
            and turn - seen[q] <= ORE_FRESH_TURNS}
    shortlist = []
    for site in visible:
        # Visibility is required to check current terrain and rover occupancy.
        if cells[site]["type"] != "empty" or site in bases or site not in distances or site in occupied:
            continue
        if distances[site] > BASE_MAX_APPROACH or home_distance.get(site, 0) < BASE_MIN_HOME_DISTANCE:
            continue
        if any(abs(q[0]-site[0]) + abs(q[1]-site[1]) <= 2 for q in enemy):
            continue
        nearby = {q: c for q, c in gold.items()
                  if max(abs(q[0]-site[0]), abs(q[1]-site[1])) <= BASE_GOLD_RADIUS}
        stock = sum(c["amount"] for c in nearby.values())
        if stock < BASE_MIN_GOLD:
            continue
        # Cheap approximation only determines which six sites merit a search.
        rough = sum(min(c["amount"], 8) / 8 * 2 * max(
                    0, home_distance[q] - max(abs(q[0]-site[0]), abs(q[1]-site[1])))
                    for q, c in nearby.items()) - distances[site] - 1
        shortlist.append((rough, site, nearby))
    shortlist.sort(key=lambda item: (-item[0], item[1]))
    # Keep a viable existing destination so the builder does not oscillate.
    previous = [item for item in shortlist if item[1] == previous_site]
    candidates = previous + [item for item in shortlist if item[1] != previous_site][:6-len(previous)]
    best = None
    for _, site, nearby in candidates:
        # Distances FROM and TO this site differ when obstacle entry costs differ.
        outward, _ = _search([site], neighbors, costs)
        inward, _ = _search([site], neighbors, costs, reverse=True)
        old_outward, _ = routes["bases"]
        saving = sum(min(c["amount"], 8) / 8 * max(
                     0, old_outward[q] + home_distance[q] - outward[q] - inward[q])
                     for q, c in nearby.items() if q in outward and q in old_outward)
        # Charge the scout's travel and one construction turn against the benefit.
        net = saving - distances[site] - 1
        if net >= BASE_MIN_NET_SAVING and (best is None or net > best["net_saving"]):
            best = {"site": site, "gold": sum(c["amount"] for c in nearby.values()),
                    "net_saving": net}
            if site == previous_site:
                return best
    return best


def calculate_actions(gameState: BotState, identity: Identity) -> ActionsByEntity:
    """Maximize deposited gold, with bounded investment and a timed return."""
    # CHUNK 5: Read the turn and update what our team remembers.
    # The supplied mars_view helper extracts the mining-game view. A pending
    # match may have no view yet; {} means we have no actions to propose.
    view = mars_view(gameState)
    if view is None:
        return {}
    team = identity["teamId"]
    turn = view["turn"]
    match = gameState["match"]
    width, height = view["map"]["width"], view["map"]["height"]
    # Keep observations between turns, but never reuse them for another match,
    # another team, a differently sized map, or a turn counter that went back.
    match_key = (match["matchId"], team, width, height)
    if _memory.get("key") != match_key or turn < _memory.get("turn", -1):
        _memory.clear()
        # cells/bases: last observed world; targets: last ore target per rover;
        # returns: rover IDs committed to delivery; seen: observation turns;
        # blocked: ore to avoid until a later turn; last_mine: previous attempts;
        # visits: tile occupancy counts, used to discourage repetitive scouting.
        _memory.update(key=match_key, cells={}, bases={}, targets={}, returns=set(),
                       seen={}, blocked={}, last_mine={}, visits={}, roster=[], scout=None,
                       base_site=None, decisions={})
    _memory["turn"] = turn
    # Default to 480 turns only when the match does not supply its own limit.
    remaining = max(0, match.get("maxTurns", 480) - turn)
    visible = set()
    for index, cell in enumerate(view["map"]["cells"]):
        if cell["type"] != "hidden":
            # The map is a flat row-by-row array. % gives the column (x);
            # integer division // gives the row (y).
            p = (index % width, index // width)
            visible.add(p)
            # dict(cell) copies the observation. Hidden tiles do not overwrite
            # memory; remembered resource amounts may therefore become stale.
            _memory["cells"][p] = dict(cell)
            _memory["seen"][p] = turn
    cells = _memory["cells"]
    # A visible tile disproves an old base observation; hidden bases stay known.
    current_bases = {(b["position"]["x"], b["position"]["y"]): b for b in view["bases"]}
    for p in list(_memory["bases"]):
        if p in visible and p not in current_bases:
            del _memory["bases"][p]
    _memory["bases"].update(current_bases)
    bases = _memory["bases"]
    # These sets contain BASE POSITIONS, not base IDs or rover positions.
    friendly = {p for p, b in bases.items() if b["teamId"] == team}
    enemy = {p for p, b in bases.items() if b["teamId"] != team}
    # Filter by authenticated team ID. Sorting makes decisions repeatable,
    # but earlier IDs also get first choice of ore and the iron-mining slot.
    workers = sorted((w for w in view["units"] if w["teamId"] == team), key=lambda w: w["unitId"])
    worker_ids = {w["unitId"] for w in workers}
    # Record birth order from observations. If we join after three already exist,
    # their exact birth order is unavailable, so sorted ID order is the fallback.
    for w in workers:
        if w["unitId"] not in _memory["roster"]:
            _memory["roster"].append(w["unitId"])
    if _memory["scout"] not in worker_ids:
        _memory["scout"] = _memory["roster"][2] if len(_memory["roster"]) >= 3 else None
    scout_id = _memory["scout"]
    # Miners choose their loads before the scout evaluates leftovers or explores.
    workers.sort(key=lambda w: (w["unitId"] == scout_id, w["unitId"]))
    _memory["decisions"] = {}
    # Drop assignments for rovers no longer in our current worker list.
    _memory["returns"].intersection_update(worker_ids)
    _memory["targets"] = {k: v for k, v in _memory["targets"].items() if k in worker_ids}
    for w in workers:
        p = (w["position"]["x"], w["position"]["y"])
        _memory["visits"][p] = _memory["visits"].get(p, 0) + 1
        # One failed mine: previous attempt was exactly one turn ago, on this
        # tile, and inventory is unchanged. Avoid that ore for 12 turns.
        # This is a heuristic: an unseen enemy claim OR losing the last ore
        # to another miner can produce this result. It is not proof of a claim.
        old = _memory["last_mine"].get(w["unitId"])
        if old and old[0] == turn - 1 and old[1] == p and old[2] == w.get("inventory", {}):
            _memory["blocked"][p] = turn + 12
    # Clear old attempts; only workers choosing mine below will add new ones.
    _memory["last_mine"] = {}

    # CHUNK 6: Turn remembered terrain into a movement graph and ore list.
    # These are default-rule costs: plain entry takes 1 turn; an obstacle takes
    # 3 including entry. Never-seen tiles and known enemy bases are excluded.
    # Other rovers do not block movement, so their occupancy is not excluded.
    costs = {p: 3 if c["type"] in ("mountain", "crater") else 1
             for p, c in cells.items() if c["type"] not in ("void", "hidden") and p not in enemy}
    neighbors = {p: [(p[0] + dx, p[1] + dy) for dx, dy in _DIRECTIONS
                     if (p[0] + dx, p[1] + dy) in costs] for p in costs}
    # Search backward from ALL known friendly bases at once. The resulting
    # home_distance[q] is the cheapest known cost from q to any friendly base.
    # & is set intersection: include only bases present in the movement graph.
    home_distance, _ = _search(sorted(friendly & costs.keys()), neighbors, costs, reverse=True)
    # Every rover starts with pass. A later decision replaces it with ONE action.
    actions = {w["unitId"]: {"type": "pass"} for w in workers}
    if not friendly:
        return actions

    def claim_ok(p):
        # Territory uses Manhattan distance: abs(dx) + abs(dy), not the diagonal
        # movement distance. Overlapping claims permit every claiming team.
        owners = {b["teamId"] for q, b in bases.items() if abs(q[0]-p[0]) + abs(q[1]-p[1]) <= 2}
        return not owners or team in owners

    # Keep stocked, apparently eligible ore with a known route back to base.
    # Claim checks use observed bases; unknown enemy bases can still surprise us.
    ores = {p: c for p, c in cells.items() if c["type"] in ("gold", "iron")
            and c.get("amount", 0) > 0 and p in home_distance and claim_ok(p)
            and _memory["blocked"].get(p, -1) <= turn}
    # CHUNK 7: Cache routes once per rover, then set investment targets.
    # Reuse these searches for scouting, construction, and mining decisions.
    routes = {w["unitId"]: _search([(w["position"]["x"], w["position"]["y"])], neighbors, costs)
              for w in workers if (w["position"]["x"], w["position"]["y"]) in costs}
    routes["bases"] = _search(sorted(friendly & costs.keys()), neighbors, costs)
    bank = view.get("teamResources", {}).get(team, {})
    bank_iron = bank.get("iron", 0)
    bank_gold = bank.get("gold", 0)
    # Current rules have no worker destruction. Count includes workers on cooldown.
    price = _spawn_price(max(0, len(workers) - 1))
    can_invest = remaining > INVEST_STOP_REMAINING
    # The first forward base gets six iron reserved before a fourth recruit.
    base_reserve = 6 if len(friendly) < 2 and can_invest else 0
    carried_iron = sum(w.get("inventory", {}).get("iron", 0) for w in workers)
    closest = {q: min((r[0].get(q, float("inf")) for uid, r in routes.items()
                       if uid != "bases"), default=float("inf")) for q in ores}
    fresh_gold = {q: c for q, c in ores.items() if c["type"] == "gold"
                  and turn - _memory["seen"][q] <= ORE_FRESH_TURNS}
    # Discount loads already carried and the gold needed to fill existing rovers.
    nearby_gold = sum(c["amount"] for q, c in fresh_gold.items()
                      if closest[q] <= FOURTH_GOLD_MAX_TRAVEL)
    existing_demand = sum(max(0, w.get("inventoryCapacity", 8) - sum(w.get("inventory", {}).values()))
                          for w in workers if w["unitId"] != scout_id)
    spare_gold = max(0, nearby_gold - existing_demand)
    nearby_iron = sum(c["amount"] for q, c in ores.items() if c["type"] == "iron"
                      and turn - _memory["seen"][q] <= ORE_FRESH_TURNS
                      and closest[q] <= IRON_MAX_TRAVEL and home_distance[q] <= IRON_MAX_RETURN)
    funding_gap = max(0, price + base_reserve - bank_iron - carried_iron)
    # A funding plan must remain valid once its iron has actually been delivered.
    iron_funds_fourth = funding_gap == 0 or nearby_iron >= funding_gap
    fourth_justified = (len(workers) == 3 and remaining >= FOURTH_MIN_REMAINING
                        and spare_gold >= FOURTH_MIN_AVAILABLE_GOLD
                        and (bank_gold >= FOURTH_BANKED_GOLD or iron_funds_fourth))
    desired_workers = 4 if fourth_justified else 3
    need_spawn = can_invest and len(workers) < desired_workers
    iron_goal = (price if need_spawn else 0) + base_reserve
    # Carried iron cannot be spent yet. Count it as expected future funding so
    # other rovers do not all gather the same outstanding iron requirement.
    iron_needed = max(0, iron_goal - bank_iron - carried_iron)
    # Empty, ready, non-mining rovers build first. The two opening recruits may
    # interrupt one mine action; a fourth waits for a rover with no active ore.
    spawn_candidates = [w for w in workers if not sum(w.get("inventory", {}).values())
                         and turn >= w.get("readyOnTurn", 0)
                         and (w["unitId"] != scout_id or len(workers) < 3)
                         and (len(workers) < 3 or (w["position"]["x"], w["position"]["y"]) not in ores)]
    spawn_id = min(spawn_candidates, key=lambda w: (
        (w["position"]["x"], w["position"]["y"]) in ores, w["unitId"]))["unitId"] if spawn_candidates else None
    build_plan = None
    if can_invest and bank_iron >= 6 and scout_id in routes and len(friendly) < MAX_BASES:
        occupied = {(w["position"]["x"], w["position"]["y"]) for w in view["units"]
                    if w["unitId"] != scout_id}
        build_plan = _base_plan(scout_id, visible, cells, bases, enemy, ores,
                                home_distance, routes, neighbors, costs, _memory["seen"], turn,
                                occupied, _memory["base_site"])
    _memory["base_site"] = build_plan["site"] if build_plan else None
    # These explain the current plan during offline inspection; no keys are logged.
    _memory["investment"] = {"spare_gold": spare_gold, "bank_gold": bank_gold,
                               "nearby_iron": nearby_iron, "fourth_justified": fourth_justified}
    # These reset EACH TURN: ore assigned so far, scout targets selected so far,
    # and whether construction or the single iron-considering slot is used.
    reserved = {}
    explore_targets = []
    construction_used = False
    iron_worker_used = False
    # Frontier destinations are known tiles with unexplored cells in vision range.
    frontiers = {}
    for p in costs:
        # First require at least one never-seen neighbor inside the map.
        if not any(0 <= p[0]+dx < width and 0 <= p[1]+dy < height
                   and (p[0]+dx, p[1]+dy) not in cells for dx, dy in _DIRECTIONS):
            continue
        # Count never-seen tiles within ordinary Manhattan vision radius 3.
        # sum counts each True as 1. This estimates the benefit of moving here;
        # it does not model the extra obstacle-neighbor vision rule.
        gain = sum(0 <= p[0]+dx < width and 0 <= p[1]+dy < height
                   and (p[0]+dx, p[1]+dy) not in cells
                   for dx in range(-3, 4) for dy in range(-3, 4) if abs(dx)+abs(dy) <= 3)
        if gain:
            frontiers[p] = gain

    # CHUNK 8: Decide separately for each rover, in the priority order below.
    # 'continue' ends THIS rover's decision and proceeds to the next rover.
    for w in workers:
        uid = w["unitId"]
        p = (w["position"]["x"], w["position"]["y"])
        inv = w.get("inventory", {})
        load = inv.get("gold", 0) + inv.get("iron", 0)
        capacity = w.get("inventoryCapacity", 8)
        # Actions during cooldown do not queue; leave this rover's pass in place.
        if turn < w.get("readyOnTurn", 0) or p not in costs:
            continue
        # One forward search gives all candidate travel times and first steps.
        distances, first = routes[uid]
        reachable_bases = friendly & distances.keys()
        home = min(reachable_bases, key=distances.get) if reachable_bases else None
        return_cost = distances.get(home, float("inf"))

        def go(target):
            # Submit only the next move, never a whole sequence. Next turn's
            # state may change the route; we calculate again when ready.
            step = first.get(target)
            if step:
                actions[uid] = {"type": "move", "direction": _DIRECTIONS[(step[0]-p[0], step[1]-p[1])]}

        # Delivery has first priority for a loaded rover already at home.
        if load and p in friendly:
            actions[uid] = {"type": "deposit"}
            _memory["returns"].discard(uid)
            continue
        # Once expansion stops, carried iron can become scoring gold. Leave time
        # for both the trade action and the eventual deposit.
        if not can_invest and inv.get("iron", 0) >= 3 and remaining > return_cost + 3:
            actions[uid] = {"type": "trade", "from": "iron"}
            continue
        # Latch the return decision so new sightings do not interrupt a delivery.
        if load >= capacity or (load and remaining <= return_cost + 5):
            _memory["returns"].add(uid)
        if not load:
            _memory["returns"].discard(uid)
        # A delivery commitment survives new ore sightings. Late iron trading
        # above is an exception because it is evaluated before this return.
        if uid in _memory["returns"]:
            if home is not None:
                go(home)
            continue
        # Too little time for useful extra work: return if loaded, otherwise idle.
        if remaining <= return_cost + 3:
            if load and home is not None:
                go(home)
            continue
        # Spawn where the selected empty rover stands; no base is required.
        # The opening can interrupt one mine action. The fourth recruit uses
        # a non-scout rover that is not standing on eligible ore.
        spendable = bank_iron - (base_reserve if len(workers) >= 3 else 0)
        if need_spawn and uid == spawn_id and spendable >= price and not construction_used:
            actions[uid] = {"type": "spawn"}
            bank_iron -= price
            construction_used = True
            _memory["decisions"][uid] = "opening recruit" if len(workers) < 3 else "funded fourth recruit"
            continue

        # CHUNK 9: The scout can become a forward builder when a measured plan
        # repays its travel/setup cost. No blind rush to the geometric center.
        if uid == scout_id and build_plan and bank_iron >= 6 and not construction_used:
            site = build_plan["site"]
            occupied = any(v["unitId"] != uid and v["position"] == {"x": site[0], "y": site[1]}
                           for v in view["units"])
            if not occupied:
                if p == site:
                    actions[uid] = {"type": "build-base"}
                    bank_iron -= 6
                    construction_used = True
                    _memory["decisions"][uid] = "build forward base"
                else:
                    go(site)
                    _memory["decisions"][uid] = "travel to forward base site"
                continue

        # CHUNK 10: Score ore and assign one target to this rover.
        # At most one rover per turn may CONSIDER iron; it can still pick gold.
        # This is not a persistent iron role and does not choose the nearest rover.
        scouting = uid == scout_id and remaining > SCOUT_STOP_REMAINING
        wants_iron = (can_invest and iron_needed > 0 and not iron_worker_used
                      and inv.get("gold", 0) == 0 and not scouting)
        if wants_iron:
            iron_worker_used = True
        best, best_value, best_amount = None, 0.0, 0
        for q, c in ores.items():
            if q not in distances:
                continue
            # The scout deliberately explores unless fresh, useful gold is close.
            if scouting and (c["type"] != "gold" or distances[q] > SCOUT_GOLD_MAX_TRAVEL
                              or turn - _memory["seen"][q] > ORE_FRESH_TURNS):
                continue
            # Do not promise later rovers ore already assigned earlier this turn.
            stock = max(0, c["amount"] - reserved.get(q, 0))
            amount = min(capacity - load, stock)
            if scouting and stock < min(SCOUT_MIN_GOLD, capacity - load):
                continue
            if c["type"] == "iron":
                if not wants_iron:
                    continue
                amount = min(amount, iron_needed)
            else:
                # Near the end, reduce the gold load target to leave time for
                # travel and delivery, rather than insisting on a full inventory.
                amount = min(amount, max(0, remaining - distances[q] - home_distance[q] - 3))
            if amount <= 0 or distances[q] + amount + home_distance[q] + 2 > remaining:
                continue
            # Estimated new ore per turn: outbound travel + one turn per ore
            # mined + return travel + one deposit action. This is a heuristic,
            # not an exact total-income calculation when already carrying ore.
            value = amount / (distances[q] + amount + home_distance[q] + 1)
            if c["type"] == "iron":
                # Treat investment iron as more valuable while expansion helps.
                value *= 2.4 if len(workers) < 3 else 1.3
            if _memory["targets"].get(uid) == q:
                # A small preference reduces constant switching between targets.
                value *= 1.15
            if q == p:
                # Prefer continuing at a deposit under this rover's feet.
                value *= 1.2
            # Prefer a current observation, but remembered ore remains a lead.
            value *= 1 / (1 + max(0, turn - _memory["seen"][q] - 8) / 80)
            # Keep the highest-scoring candidate seen in this loop.
            if value > best_value:
                best, best_value, best_amount = q, value, amount
        if best is not None:
            # Reserve an expected LOAD, not just this turn's 1 kg mine action.
            # Reservations are local planning only; opponents can take the ore.
            reserved[best] = reserved.get(best, 0) + best_amount
            _memory["targets"][uid] = best
            _memory["decisions"][uid] = "scout collects nearby gold" if scouting else "collect " + ores[best]["type"]
            if ores[best]["type"] == "iron":
                iron_needed = max(0, iron_needed - best_amount)
            if best == p:
                actions[uid] = {"type": "mine"}
                _memory["last_mine"][uid] = (turn, p, dict(inv))
            else:
                go(best)
            continue

        # CHUNK 11: No suitable ore target: deliver a partial load or scout.
        # Deliver partial loads when reachable useful ore has run out.
        if load and home is not None:
            _memory["returns"].add(uid)
            go(home)
            continue
        # Explore separate frontiers with a modest preference toward the center.
        target, value = None, 0.0
        cx, cy = (width-1)/2, (height-1)/2
        for q, gain in frontiers.items():
            if q not in distances or q == p or distances[q] + home_distance.get(q, 1000) + 4 >= remaining:
                continue
            # Keep this turn's chosen scouting destinations spread apart. This
            # measures target separation, not actual rover-to-rover distance.
            separation = min((max(abs(q[0]-v[0]), abs(q[1]-v[1])) for v in explore_targets), default=10)
            center = max(abs(q[0]-cx), abs(q[1]-cy)) / max(width, height)
            # More discoveries and separation help; travel time and distance
            # from the center hurt. Center bias is modest, not a center rush.
            score = gain * (0.4 + min(separation, 6)/6) / (distances[q] + 3) / (1 + center)
            # Prefer destinations we have not repeatedly occupied.
            score /= 1 + 0.2 * _memory["visits"].get(q, 0)
            if score > value:
                target, value = q, score
        # If no frontier is selected, revisit old observations for respawned ore.
        # This fallback can also happen when some unexplored area is unreachable.
        if target is None:
            candidates = [q for q in distances if q != p and distances[q] + home_distance.get(q, 1000) + 4 < remaining]
            if candidates:
                # Prefer older observations with shorter trips; discount cells
                # near another destination already selected in this turn.
                target = max(candidates, key=lambda q: (turn-_memory["seen"].get(q, 0)) / (distances[q]+5)
                             / (1 + sum(max(abs(q[0]-v[0]), abs(q[1]-v[1])) < 4 for v in explore_targets)))
        if target is not None:
            explore_targets.append(target)
            go(target)
            _memory["decisions"][uid] = "scout frontier" if scouting else "explore for resources"
    # The client submits this dictionary. It maps rover IDs to one action each.
    return actions


# Game helpers. Feel free to use, discard, or modify.
# Fill in the strategy section above.

def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _cell(grid: CoordinateGrid | GameMap, point: Coordinate):
    x, y = point["x"], point["y"]
    if not _is_integer(x) or not _is_integer(y):
        return None
    if not (0 <= x < grid["width"] and 0 <= y < grid["height"]):
        return None

    row_start = y * grid["width"]
    return grid["cells"][row_start + x]


def is_claimed(state: TerritoryState | MarsView, coordinate: Coordinate) -> list[str]:
    """Return all known claiming teams, including overlapping claims."""
    teams: list[str] = []

    for base in state["bases"]:
        position = base["position"]
        distance = abs(position["x"] - coordinate["x"]) + abs(position["y"] - coordinate["y"])
        if distance <= 2 and base["teamId"] not in teams:
            teams.append(base["teamId"])

    return teams


def is_base(state: TerritoryState | MarsView, coordinate: Coordinate) -> Optional[str]:
    for base in state["bases"]:
        if base["position"] == coordinate:
            return base["teamId"]
    return None


def is_ore(state: OreState | MarsView, coordinate: Coordinate) -> Optional[Resource]:
    cell = _cell(state["map"], coordinate)
    if cell and cell["type"] in ("gold", "iron") and cell.get("amount", 0) > 0:
        return cell["type"]
    return None


def can_move_onto(
    state: MovementState | MarsView,
    coordinate: Coordinate,
    team_id: Optional[str] = None,
) -> bool:
    """Unknown terrain and enemy bases are excluded; workers do not block."""
    cell = _cell(state["map"], coordinate)
    if cell is None or cell["type"] in ("hidden", "void"):
        return False

    base_owner = is_base(state, coordinate)
    return base_owner is None or base_owner == team_id


def path_to(
    grid: CoordinateGrid | GameMap,
    start: Coordinate,
    end: Coordinate,
    *,
    bases: Sequence[BaseLocation] = (),
    team_id: Optional[str] = None,
    obstacle_cost: int = 3,
) -> Optional[list[Coordinate]]:
    """Cost until ready at destination; start must be ready; None if unreachable."""
    if not _is_integer(obstacle_cost) or obstacle_cost < 1:
        return None

    state: MovementState = {"map": grid, "bases": bases}
    if not can_move_onto(state, start, team_id) or not can_move_onto(state, end, team_id):
        return None

    origin = (start["x"], start["y"])
    target = (end["x"], end["y"])
    distances = {origin: 0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0, origin)]

    while queue:
        distance, position = heappop(queue)
        if distance != distances[position]:
            continue
        if position == target:
            break

        x, y = position
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if not dx and not dy:
                    continue
                point: Coordinate = {"x": x + dx, "y": y + dy}
                if not can_move_onto(state, point, team_id):
                    continue

                neighbor = (point["x"], point["y"])
                cell = _cell(grid, point)
                movement_cost = obstacle_cost if cell and cell["type"] in ("mountain", "crater") else 1
                candidate = distance + movement_cost
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = position
                    heappush(queue, (candidate, neighbor))

    if target not in distances:
        return None

    path: list[Coordinate] = []
    position = target
    while position != origin:
        path.append({"x": position[0], "y": position[1]})
        position = previous[position]

    return list(reversed(path))


if __name__ == "__main__":
    print("Gold mining bot: recruit, explore, mine, and bank before the deadline.")
    if "LUCID_API_BASE_URL" not in os.environ:
        print(f"LUCID_API_BASE_URL is not set; using {API_BASE_URL}.")

    try:
        BotClient(API_KEY, API_BASE_URL).connect_and_run_bot(calculate_actions)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(error)
        raise SystemExit(1)
