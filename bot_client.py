import json
import sys
import http.client
import time
import ssl
from uuid import uuid4
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit, unquote
from base64 import b64encode
from collections.abc import Callable, Generator, Iterator, Sequence
from typing import Literal, Optional, TypedDict, Union, cast


class Coordinate(TypedDict):
    x: int
    y: int


Resource = Literal["gold", "iron"]
Direction = Literal["north", "east", "south", "west", "north-east", "south-east", "south-west", "north-west"]
PlanetName = Literal["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]


class Inventory(TypedDict):
    gold: int
    iron: int


class TerrainCell(TypedDict):
    type: Literal["empty", "mountain", "hidden", "void"]


class CraterFormation(TypedDict):
    x: int
    y: int
    size: int


class _CraterOptions(TypedDict, total=False):
    formation: CraterFormation


class CraterCell(_CraterOptions):
    type: Literal["crater"]


class OreCell(TypedDict):
    type: Resource
    amount: int


class UnitCell(TypedDict):
    type: Literal["unit"]
    unitId: str
    teamId: str


MapCell = Union[TerrainCell, CraterCell, OreCell, UnitCell]


class _GridCellOptions(TypedDict, total=False):
    amount: int


class GridCell(_GridCellOptions):
    type: str


class CoordinateGrid(TypedDict):
    width: int
    height: int
    cells: Sequence[GridCell]


class GameMap(TypedDict):
    width: int
    height: int
    cells: Sequence[MapCell]


class BaseLocation(TypedDict):
    teamId: str
    position: Coordinate


class Base(BaseLocation):
    baseId: str


class TerritoryState(TypedDict):
    bases: Sequence[BaseLocation]


class OreState(TypedDict):
    map: Union[CoordinateGrid, GameMap]


class MovementState(TerritoryState, OreState):
    pass


class _UnitOptions(TypedDict, total=False):
    inventory: Inventory


class Unit(_UnitOptions):
    unitId: str
    teamId: str
    position: Coordinate
    inventoryCapacity: int
    readyOnTurn: int


class Notice(TypedDict):
    entityId: str
    error: Literal["movement_interrupted"]
    note: str


class BonkEffect(TypedDict):
    type: Literal["bonk"]
    coordinate: Coordinate
    entityIds: list[str]
    note: str


class DepositEffect(TypedDict):
    type: Literal["resource_deposit"]
    coordinate: Coordinate
    baseId: str
    resource: Resource
    amount: int
    total: int


class HighGoldSettings(TypedDict):
    width: int
    height: int
    min: int
    max: int
    spawnPercent: int


class _MarsViewOptions(TypedDict, total=False):
    baseResourceDeposits: dict[str, Inventory]


class MarsView(_MarsViewOptions):
    turn: int
    map: GameMap
    units: list[Unit]
    bases: list[Base]
    scores: dict[str, int]
    teamResources: dict[str, Inventory]
    highGoldArea: HighGoldSettings
    warnings: list[Notice]
    info: list[Notice]
    effects: list[Union[BonkEffect, DepositEffect]]


class Position(TypedDict):
    x: float
    y: float


class SolarPlanet(TypedDict):
    name: PlanetName
    position: Position
    orbitRadius: float


class Orbit(TypedDict):
    planet: PlanetName
    angle: float


class SolarRocket(TypedDict):
    teamId: str
    position: Position
    previousPosition: Position
    heading: float
    target: Optional[PlanetName]
    targetPosition: Optional[Position]
    orbit: Optional[Orbit]


class _SolarViewOptions(TypedDict, total=False):
    entityId: str


class SolarView(_SolarViewOptions):
    turn: int
    planets: list[SolarPlanet]
    rockets: list[SolarRocket]


class MoveAction(TypedDict):
    type: Literal["move"]
    direction: Direction


class SimpleAction(TypedDict):
    type: Literal["mine", "pass", "deposit", "spawn", "build-base"]


# The wire field is named "from", a Python keyword.
TradeAction = TypedDict("TradeAction", {"type": Literal["trade"], "from": Resource})


class FollowPlanetAction(TypedDict):
    type: PlanetName


class FlyAction(TypedDict):
    type: Literal["move"]
    x: float
    y: float


Action = Union[MoveAction, SimpleAction, TradeAction, FollowPlanetAction, FlyAction]


class _MatchOptions(TypedDict, total=False):
    cancellationReason: str
    turnAdvanceMode: str
    rulesVersion: str


class TeamDescriptor(TypedDict):
    id: str
    displayName: str
    botType: Union[Literal[False], str]


class MatchSummary(_MatchOptions):
    teams: list[TeamDescriptor]
    matchId: str
    status: str
    turn: int
    maxTurns: int


class _StateOptions(TypedDict, total=False):
    actionRevisions: dict[str, int]
    view: Union[MarsView, SolarView]
    turnEndsAtMs: float


class BotState(_StateOptions):
    match: MatchSummary
    serverTimeMs: float


class _TurnOptions(TypedDict, total=False):
    turnEndsAtMs: float


class TurnEvent(_TurnOptions):
    turn: int
    view: Union[MarsView, SolarView]
    serverTimeMs: float


class MatchStartingEvent(TypedDict):
    message: str
    match: MatchSummary
    serverTimeMs: float
    startsAtMs: float


class MatchEndedEvent(TypedDict):
    match: MatchSummary
    serverTimeMs: float


class AcceptedAction(TypedDict):
    acceptedForTurn: int


class ActionResponse(TypedDict, total=False):
    accepted: list[AcceptedAction]
    submissionId: str
    state: Optional[BotState]


class Identity(TypedDict):
    teamId: str
    displayName: str


ActionsByEntity = dict[str, Action]
CalculateActions = Callable[[BotState, Identity], ActionsByEntity]
ServerEvent = tuple[str, object]


def mars_view(state: BotState) -> Optional[MarsView]:
    view = state.get("view")
    return view if view is not None and "units" in view else None


def solar_view(state: BotState) -> Optional[SolarView]:
    view = state.get("view")
    return view if view is not None and "rockets" in view else None


REQUEST_TIMEOUT_SECONDS = 5
STREAM_IDLE_TIMEOUT_SECONDS = 10
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 0.25


def is_transient(error: BaseException) -> bool:
    if isinstance(error, ApiError):
        return error.status in (408, 429, 500, 502, 503, 504)
    if isinstance(error, ssl.SSLCertVerificationError):
        return False
    if isinstance(error, urllib.error.URLError) and isinstance(
        error.reason, ssl.SSLCertVerificationError
    ):
        return False
    return isinstance(error, (OSError, http.client.HTTPException))


class BotClient:
    def __init__(self, api_key: str, api_base_url: str) -> None:
        if not api_key.strip() or api_key == "replace-with-api-key":
            raise ValueError("Set API_KEY to your team API key before starting the bot.")
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")
        endpoint = urlsplit(self.api_base_url)
        if endpoint.scheme not in ("http", "https") or endpoint.hostname is None:
            raise ValueError("The competition URL must use HTTP or HTTPS and include a host.")
        self.host = endpoint.hostname
        self.port = endpoint.port
        self.secure = endpoint.scheme == "https"
        self.http: Optional[http.client.HTTPConnection] = None
        self.stream_http: Optional[http.client.HTTPConnection] = None
        self.http_absolute_url = False
        self.http_proxy_headers: dict[str, str] = {}
        self.revisions: dict[str, int] = {}
        self.bot_session_token: Optional[str] = None
        self.bot_session_revision: Optional[str] = None
        self.bot_match_id: Optional[str] = None
        self.events: Optional[Generator[ServerEvent, None, None]] = None
        self.events_match_id: Optional[str] = None

    def _connection(self, timeout: float) -> tuple[http.client.HTTPConnection, bool, dict[str, str]]:
        proxy_url = urllib.request.getproxies().get("https" if self.secure else "http")
        if proxy_url and not urllib.request.proxy_bypass(f"{self.host}:{self.port}" if self.port else self.host):
            proxy = urlsplit(proxy_url if "://" in proxy_url else "http://" + proxy_url)
            if proxy.scheme != "http" or proxy.hostname is None:
                raise ValueError("Use an HTTP proxy URL in HTTP_PROXY or HTTPS_PROXY.")
            headers: dict[str, str] = {}
            if proxy.username is not None:
                credentials = unquote(proxy.username) + ":" + unquote(proxy.password or "")
                headers["Proxy-Authorization"] = "Basic " + b64encode(credentials.encode()).decode()
            if self.secure:
                connection = http.client.HTTPSConnection(proxy.hostname, proxy.port or 80, timeout=timeout)
                connection.set_tunnel(self.host, self.port or 443, headers=headers)
                return connection, False, {}
            return http.client.HTTPConnection(proxy.hostname, proxy.port or 80, timeout=timeout), True, headers
        factory = http.client.HTTPSConnection if self.secure else http.client.HTTPConnection
        return factory(self.host, self.port, timeout=timeout), False, {}

    def _open(self, request: urllib.request.Request, connection: http.client.HTTPConnection,
              absolute_url: bool, proxy_headers: dict[str, str]) -> http.client.HTTPResponse:
        headers = {**dict(request.header_items()), **proxy_headers}
        connection.request(request.get_method(), request.full_url if absolute_url else request.selector,
                           request.data, headers)
        response = connection.getresponse()
        if response.status >= 400:
            try:
                body = response.read().decode("utf-8")
            finally:
                response.close()
            raise ApiError(response.status, response.reason, body)
        return response

    def _open_json(self, request: urllib.request.Request) -> http.client.HTTPResponse:
        if self.http is None:
            self.http, self.http_absolute_url, self.http_proxy_headers = self._connection(REQUEST_TIMEOUT_SECONDS)
        return self._open(request, self.http, self.http_absolute_url, self.http_proxy_headers)

    def _open_stream(self, request: urllib.request.Request) -> http.client.HTTPResponse:
        self.stream_http, absolute_url, proxy_headers = self._connection(STREAM_IDLE_TIMEOUT_SECONDS)
        return self._open(request, self.stream_http, absolute_url, proxy_headers)

    def close_http(self) -> None:
        connection, self.http = self.http, None
        if connection is not None:
            connection.close()

    def close(self) -> None:
        try:
            self.close_events()
        finally:
            self.close_http()

    def request(self, path: str, method: str = "GET", body: object = None,
                headers: Optional[dict[str, str]] = None, recovering: bool = False) -> object:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_base_url}{path}",
            data=data,
            method=method,
            headers={
                "x-api-key": self.api_key,
                **({"x-bot-session": self.bot_session_token} if self.bot_session_token else {}),
                "content-type": "application/json",
                **(headers or {}),
            },
        )

        # Retry writes only when a match ID and idempotency token pin the request.
        keyed_write = (
            method == "PUT" and "/current/" not in path
            and isinstance(body, dict)
            and bool(cast(dict[str, object], body).get("submissionId"))
        )
        bound_registration = (
            path == "/api/team/matches/current/register" and headers is not None
            and "x-bot-match-id" in headers and self.bot_session_token is not None
        )
        attempts = MAX_ATTEMPTS if method == "GET" or path == "/api/auth/api-key" or keyed_write or bound_registration else 1
        for attempt in range(attempts):
            try:
                with self._open_json(req) as response:
                    if path == "/api/team/matches/current/register":
                        self.bot_session_token = response.headers.get("x-bot-session") or self.bot_session_token
                    self.bot_session_revision = response.headers.get("x-bot-session-revision") or self.bot_session_revision
                    return json.loads(response.read().decode("utf-8"))
            except (ApiError, OSError, http.client.HTTPException) as error:
                if (isinstance(error, ApiError) and error.status == 409
                    and 'bot_session_missing' in error.body and not recovering
                    and self.bot_match_id and self.bot_session_token
                    and path != "/api/team/matches/current/register"):
                    self.request("/api/team/matches/current/register", method="POST",
                                 headers={"x-bot-match-id": self.bot_match_id, "x-bot-session-recover": "1"}, recovering=True)
                    return self.request(path, method, cast(object, body), headers, recovering=True)
                if not is_transient(error):
                    if not isinstance(error, ApiError):
                        self.close_http()
                    if isinstance(error, ApiError) and error.status == 404 and "/matches/" not in path:
                        raise RuntimeError(f"Cannot reach the competition API at {self.api_base_url}; check LUCID_API_BASE_URL.") from error
                    raise
                self.close_http()
                if attempt + 1 == attempts:
                    raise RuntimeError(
                        f"Cannot reach {self.api_base_url}; check the URL and network connection ({attempts} attempt(s))."
                    ) from error
            time.sleep(RETRY_DELAY_SECONDS * 2 ** attempt)
        raise RuntimeError("Request attempts exhausted.")

    def validate_api_key(self) -> Identity:
        return cast(Identity, self.request("/api/auth/api-key", method="POST"))

    def current_match_id(self) -> str:
        return self.get_state()["match"]["matchId"]

    def register_team_to_match(self) -> BotState:
        try:
            match_id = self.current_match_id()
        except ApiError as error:
            if not error.is_match_unavailable():
                raise
            announced = ""
            while True:
                response = cast(dict[str, object], self.request("/api/team/matches/current/register", method="POST"))
                if "match" in response:
                    match_id = self.current_match_id()
                    break
                if "queue" not in response:
                    raise RuntimeError("Invalid registration response.")
                queue = cast(dict[str, object], response["queue"])
                waiting = ("Waiting for a private practice match. Create one on the website; keep this bot running."
                           if queue.get("autoQueue") is False else "Waiting in PvP queue; keep this bot running.")
                if announced != waiting:
                    print(waiting)
                    announced = waiting
                time.sleep(1)
        self.bot_match_id = match_id
        self.bot_session_token = self.bot_session_token or str(uuid4())
        state = cast(
            BotState,
            self.request("/api/team/matches/current/register", method="POST",
                         headers={"x-bot-match-id": match_id, **({"x-bot-session-revision": self.bot_session_revision} if self.bot_session_revision else {})}),
        )
        print(f"registered for match {state['match']['matchId']}")

        if state["match"]["status"] == "pending":
            print("waiting for all teams to join")
        else:
            print("All players joined, starting match")

        return state

    def wait_for_match_start(
        self, events: Iterator[ServerEvent], state: BotState
    ) -> BotState:
        return self.wait_for_progress(events, state, starting=True)

    def wait_for_progress(
        self, events: Iterator[ServerEvent], state: BotState, starting: bool
    ) -> BotState:
        attempt = 0
        while attempt < MAX_ATTEMPTS:
            idle = False
            try:
                for event_name, event_data in events:
                    if event_name == "match_ended":
                        self.close_events()
                        ended = cast(MatchEndedEvent, event_data)
                        return {**state, "match": ended["match"], "serverTimeMs": ended["serverTimeMs"]}
                    if event_name == "match_starting":
                        start = cast(MatchStartingEvent, event_data)
                        if start["match"]["turn"] < state["match"]["turn"]:
                            continue
                        state = self.apply_match_starting_event(state, start)
                        if starting:
                            return state
                    if event_name == "turn":
                        turn = cast(TurnEvent, event_data)
                        if turn["turn"] > state["match"]["turn"]:
                            next_state = self.state_from_turn_event(state, turn)
                            if is_terminal(next_state):
                                self.close_events()
                            return next_state
            except (ApiError, OSError, http.client.HTTPException) as error:
                if not is_transient(error):
                    self.close_events()
                    raise
                idle = isinstance(error, TimeoutError)
            except Exception:
                self.close_events()
                raise
            self.close_events()
            latest = self.get_state(state["match"]["matchId"])
            if (is_terminal(latest) or latest["match"]["turn"] > state["match"]["turn"]
                    or (starting and latest["match"]["status"] != "pending")):
                return latest
            state = latest
            # A quiet pending lobby is healthy when state polling still succeeds.
            attempt = (
                0 if idle and starting and state["match"]["status"] == "pending"
                else attempt + 1
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS * 2 ** max(attempt - 1, 0))
                events = self.match_events(state)
        raise RuntimeError(
            f"Match {state['match']['matchId']} event stream made no progress after {MAX_ATTEMPTS} attempts."
        )

    def match_events(self, state: BotState) -> Generator[ServerEvent, None, None]:
        match_id = state["match"]["matchId"]
        if self.events is None or self.events_match_id != match_id:
            self.close_events()
            target = quote(match_id, safe="")
            self.events = self.stream_events(f"/api/events/matches/{target}?afterTurn={state['match']['turn']}")
            self.events_match_id = match_id
        return self.events

    def close_events(self) -> None:
        events, self.events = self.events, None
        self.events_match_id = None
        if events is not None:
            events.close()

    def stream_events(self, path: str) -> Generator[ServerEvent, None, None]:
        req = urllib.request.Request(
            f"{self.api_base_url}{path}",
            method="GET",
            headers={
                "x-api-key": self.api_key,
                **({"x-bot-session": self.bot_session_token} if self.bot_session_token else {}),
                "accept": "text/event-stream",
            },
        )

        try:
            with self._open_stream(req) as response:
                event_name = "message"
                data_lines: list[str] = []

                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()

                    if line == "":
                        if data_lines:
                            yield event_name, json.loads("\n".join(data_lines))
                            event_name = "message"
                            data_lines = []
                        continue

                    if line.startswith(":"):
                        continue

                    if line.startswith("event:"):
                        event_name = line.removeprefix("event:").strip()
                    elif line.startswith("data:"):
                        data_lines.append(line.removeprefix("data:").strip())
        finally:
            if self.stream_http is not None:
                self.stream_http.close()
                self.stream_http = None

    def replace_actions(
        self, actions: ActionsByEntity, match_id: Optional[str] = None
    ) -> ActionResponse:
        submissions: list[dict[str, object]] = []

        for entity_id, action in actions.items():
            revision = self.revisions.get(entity_id, 0) + 1
            self.revisions[entity_id] = revision
            submissions.append(
                {
                    "entityId": entity_id,
                    "revision": revision,
                    "action": action,
                }
            )

        return cast(
            ActionResponse,
            self.request(
                (
                    f"/api/team/matches/{quote(match_id, safe='')}/action"
                    if match_id else "/api/team/matches/current/actions"
                ),
                method="PUT",
                body={"submissionId": str(uuid4()), "submissions": submissions},
            ),
        )

    def advance_entity_actions(self, match_id: Optional[str] = None) -> ActionResponse:
        return cast(
            ActionResponse,
            self.request(
                (
                    f"/api/team/matches/{quote(match_id, safe='')}/action"
                    if match_id else "/api/team/matches/current/actions"
                ),
                method="PUT",
                body={"submissionId": str(uuid4()), "advanceOnly": True},
            ),
        )

    def get_state(self, match_id: Optional[str] = None) -> BotState:
        target = quote(match_id, safe="") if match_id else "current"
        return cast(BotState, self.request(f"/api/team/matches/{target}/state"))

    def wait_for_next_turn(
        self, events: Iterator[ServerEvent], state: BotState
    ) -> BotState:
        return self.wait_for_progress(events, state, starting=False)

    def apply_match_starting_event(
        self, state: BotState, event_data: MatchStartingEvent
    ) -> BotState:
        next_state = cast(BotState, dict(state))
        next_state["match"] = event_data["match"]
        next_state["serverTimeMs"] = event_data["serverTimeMs"]
        next_state["turnEndsAtMs"] = event_data["startsAtMs"]
        return next_state

    def state_from_turn_event(self, state: BotState, event_data: TurnEvent) -> BotState:
        match = cast(MatchSummary, dict(state["match"]))
        match["turn"] = event_data["turn"]
        match["status"] = "completed" if event_data["turn"] >= match["maxTurns"] else "active"

        next_state: BotState = {
            "match": match,
            "view": event_data["view"],
            "serverTimeMs": event_data["serverTimeMs"],
        }

        if "turnEndsAtMs" in event_data:
            next_state["turnEndsAtMs"] = event_data["turnEndsAtMs"]

        return next_state

    def recover_state(self, match_id: str) -> BotState:
        while True:
            try:
                return self.get_state(match_id)
            except Exception as error:
                if isinstance(error, ApiError) and error.is_auth_error():
                    raise
                print(f"Cannot refresh match state: {error}; retrying.")
                time.sleep(RETRY_DELAY_SECONDS)

    def wait_for_live_progress(self, state: BotState, starting: bool = False) -> BotState:
        while True:
            try:
                return self.wait_for_progress(self.match_events(state), state, starting)
            except Exception as error:
                if isinstance(error, ApiError) and error.is_auth_error():
                    raise
                print(f"Cannot read turn updates: {error}; reconnecting.")
                latest = self.recover_state(state["match"]["matchId"])
                if is_terminal(latest) or latest["match"]["turn"] > state["match"]["turn"] or (
                    starting and latest["match"]["status"] != "pending"
                ):
                    return latest
                state = latest
                time.sleep(RETRY_DELAY_SECONDS)

    def connect_and_run_bot(self, calculate_actions: CalculateActions) -> None:
        try:
            identity = self.validate_api_key()
            try:
                gameState = self.register_team_to_match()
            except ApiError as error:
                if not error.is_pvp_disabled():
                    raise
                print("No match assigned. PvP practice is currently disabled.\nCreate a practice match on the website or ask an organizer to assign one,\nthen run your bot again.", file=sys.stderr)
                raise SystemExit(1) from None
            self.revisions = dict(gameState.get("actionRevisions", {}))
            # One run belongs to this match, even if another becomes current.
            match_id = gameState["match"]["matchId"]
            isClientActionMatch = gameState["match"].get("turnAdvanceMode") == "client-action"

            if gameState["match"]["status"] == "pending":
                gameState = self.wait_for_live_progress(gameState, starting=True)

            refresh_after_late_submission = False
            while not is_terminal(gameState):
                if not isClientActionMatch:
                    latest = self.recover_state(match_id) if refresh_after_late_submission else gameState
                    refresh_after_late_submission = False
                    if is_terminal(latest) or latest["match"]["turn"] > gameState["match"]["turn"]:
                        gameState = latest
                    else:
                        gameState = self.wait_for_live_progress(gameState)

                matchStatus = gameState["match"]["status"]
                if matchStatus != "active":
                    print(gameState["match"].get("cancellationReason") or f"match {gameState['match']['matchId']} is {matchStatus}; stopping bot")
                    return

                view_received_at = time.monotonic()
                deadline = gameState.get("turnEndsAtMs")
                if deadline is not None and gameState["serverTimeMs"] >= deadline:
                    latest = self.recover_state(match_id)
                    if is_terminal(latest):
                        gameState = latest
                        break
                    if latest["match"]["turn"] > gameState["match"]["turn"]:
                        print(f"Skipping stale turn {gameState['match']['turn']}; using turn {latest['match']['turn']}.")
                        gameState = latest
                        view_received_at = time.monotonic()
                turn = gameState["match"]["turn"]
                try:
                    actions = calculate_actions(gameState, identity)
                except Exception as error:
                    print(f"Strategy failed on turn {turn}: {error}; submitting no actions.")
                    actions = {}

                response: ActionResponse
                try:
                    response = self.replace_actions(actions, match_id)
                except Exception as error:
                    if isinstance(error, ApiError) and error.is_auth_error():
                        raise
                    latest = self.recover_state(match_id)
                    if is_terminal(latest):
                        gameState = latest
                        break
                    print(f"Action submission failed on turn {turn}: {error}; submitting no actions.")
                    try:
                        response = self.replace_actions({}, match_id)
                    except Exception as fallback_error:
                        if isinstance(fallback_error, ApiError) and fallback_error.is_auth_error():
                            raise
                        print(f"No-action submission failed: {fallback_error}; continuing.")
                        response = {}
                accepted = response.get("accepted", [])
                late = any(action["acceptedForTurn"] > turn + 1 for action in accepted)
                deadline = gameState.get("turnEndsAtMs")
                server_now_ms = gameState["serverTimeMs"] + (time.monotonic() - view_received_at) * 1000
                refresh_after_late_submission = late or (deadline is not None and server_now_ms >= deadline)
                if refresh_after_late_submission:
                    print(f"Actions for turn {turn} arrived late; shorten your strategy calculation.")
                else:
                    print(f"submitted turn {turn}, actions: {actions}")

                if isClientActionMatch:
                    gameState = response.get("state") or self.recover_state(match_id)
            print(gameState["match"].get("cancellationReason") or f"match {match_id} is {gameState['match']['status']}; stopping bot")
        finally:
            self.close()


def is_terminal(state: BotState) -> bool:
    return state["match"]["status"] in ("completed", "cancelled")


class ApiError(RuntimeError):
    def __init__(self, status: int, reason: str, body: str) -> None:
        self.status = status
        self.reason = reason
        self.body = body
        if self.is_match_unavailable():
            message = "No current match is available; join a match on the competition page."
        elif status == 409 and self.is_auth_error():
            message = "Another bot session is using this API key; stop that bot before starting this one."
        elif self.is_auth_error():
            message = "The API key was rejected; check API_KEY in your environment."
        elif status == 404:
            message = "No current match is available; join a match on the competition page."
        else:
            try:
                message = str(json.loads(body)["error"]["message"])
            except (ValueError, KeyError, TypeError):
                message = f"The server rejected the request ({status} {reason})."
        super().__init__(message)

    def is_auth_error(self) -> bool:
        if self.status in (401, 403):
            return not self.is_match_unavailable()
        try:
            return self.status == 409 and json.loads(self.body)["error"]["code"] == "bot_session_conflict"
        except (ValueError, KeyError, TypeError):
            return False

    def is_pvp_disabled(self) -> bool:
        if self.status != 409:
            return False
        try:
            return json.loads(self.body)["error"]["code"] == "pvp_disabled"
        except (ValueError, KeyError, TypeError):
            return False

    def is_match_unavailable(self) -> bool:
        if self.status != 403:
            return False
        try:
            detail = json.loads(self.body)["error"]
            return (
                detail.get("code") == "forbidden"
                and detail.get("message") == "Team is not in an active match."
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            return False
