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
                       seen={}, blocked={}, last_mine={}, visits={})
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
    # CHUNK 7: Set investment targets and list possible scouting destinations.
    # teamResources is keyed by team ID; a missing balance defaults to zero.
    bank_iron = view.get("teamResources", {}).get(team, {}).get("iron", 0)
    # Current rules have no worker destruction. Count includes workers on cooldown.
    price = _spawn_price(max(0, len(workers) - 1))
    gold_stock = sum(c["amount"] for c in ores.values() if c["type"] == "gold")
    # STRATEGY CHOICES, not game rules: normally 3 rovers; allow 4 before turn
    # 160 if remembered eligible gold >= 24. Never remove an existing fourth.
    desired_workers = 4 if turn < 160 and gold_stock >= 24 else 3
    # Stop investing in the last 180 turns (turn 300 onward in a 480-turn game).
    can_invest = remaining > 180
    need_spawn = can_invest and len(workers) < desired_workers
    # Save for a forward base after the two opening recruits.
    iron_goal = (price if need_spawn else 0) + (6 if can_invest and len(friendly) < 3 else 0)
    # Carried iron cannot be spent yet. Count it as expected future funding so
    # other rovers do not all gather the same outstanding iron requirement.
    carried_iron = sum(w.get("inventory", {}).get("iron", 0) for w in workers)
    iron_needed = max(0, iron_goal - bank_iron - carried_iron)
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
        distances, first = _search([p], neighbors, costs)
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
        # Spawn WHERE THIS ROVER STANDS. No base or journey home is required.
        # Requiring an empty inventory is our policy, not a server requirement.
        # Spawn is checked before mining; the current code can interrupt a miner.
        if need_spawn and bank_iron >= price and not construction_used and not load:
            actions[uid] = {"type": "spawn"}
            bank_iron -= price
            construction_used = True
            continue

        # CHUNK 9: Opportunistic construction at the CURRENT tile, not a plan
        # that sends a designated builder to a previously selected destination.
        # Build only at a safe plain site with nearby gold and substantial travel
        # savings. Limiting one construction per turn avoids overspending the bank.
        if can_invest and len(workers) >= 3 and len(friendly) < 3 and bank_iron >= 6 and not construction_used:
            # Count all visible rovers on the tile, including opposing rovers.
            alone = sum(v["position"] == w["position"] for v in view["units"]) == 1
            safe = all(abs(q[0]-p[0]) + abs(q[1]-p[1]) > 2 for q in enemy)
            near_gold = [(q, c) for q, c in ores.items() if c["type"] == "gold"
                         and max(abs(q[0]-p[0]), abs(q[1]-p[1])) <= 4]
            # Rough score: up to 8 ore per deposit times its potential reduction
            # in return travel. This compares q->old base with p->q, so entry
            # terrain can make it approximate; it is not a proven payback model.
            saving = sum(min(c["amount"], 8) * max(0, home_distance[q] - distances.get(q, 1000))
                         for q, c in near_gold)
            if cells[p]["type"] == "empty" and p not in bases and alone and safe and return_cost >= 6 and saving >= 55:
                actions[uid] = {"type": "build-base"}
                bank_iron -= 6
                construction_used = True
                continue

        # CHUNK 10: Score ore and assign one target to this rover.
        # At most one rover per turn may CONSIDER iron; it can still pick gold.
        # This is not a persistent iron role and does not choose the nearest rover.
        wants_iron = can_invest and iron_needed > 0 and not iron_worker_used and inv.get("gold", 0) == 0
        if wants_iron:
            iron_worker_used = True
        best, best_value, best_amount = None, 0.0, 0
        for q, c in ores.items():
            if q not in distances:
                continue
            # Do not promise later rovers ore already assigned earlier this turn.
            stock = max(0, c["amount"] - reserved.get(q, 0))
            amount = min(capacity - load, stock)
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
