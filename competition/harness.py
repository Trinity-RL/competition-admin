#!/usr/bin/env python3
"""
TMRC Competition Harness

Runs inside the participant's Docker image via:
  docker run --entrypoint python3 <image> /competition/harness.py

Starts Xvfb, launches the game, connects to TMInterface, and drives one
attempt per competition map with the participant's
    decide(game_state, wall_distances) -> {left, right, accelerate, brake}
from /agent/agent.py: the same interface, observations and decision rate as
the TMRC framework (TMRC/agent/env.py), so a driver tested there behaves the
same here.

Rules (keep TMRC/docs/11-competition.md in sync):
  - one decision every STEP_MS (50 ms) of game time, starting at race time 0;
    nothing is asked during the countdown
  - RACE_TIME_LIMIT_MS (5 minutes) of *game* time per map, so a slow decide()
    or a busy runner doesn't cost race time
  - no "stuck" rule: the attempt runs until the finish or the time limit
  - if decide() raises or returns something unusable, the car accelerates for
    that step and the error is counted
  - agent.py is imported fresh for each map, with TM_MAP_NAME set to that map,
    so module-level state starts clean and track_features.py loads the right map

Environment variables (set by evaluate.yml):
  COMPETITION_MAPS   JSON array: [{id, name, file}, ...]
  PARTICIPANT        GitHub username
  RUN_UUID           UUID for this evaluation run
  REPOSITORY         GitHub repo (org/submission-username)
  COMMIT_SHA         Git commit that triggered this run
Optional:
  HARNESS_GAME_SPEED        simulation speed multiplier (default 5; doesn't change results)
  HARNESS_MAP_WALL_LIMIT_S  real-time safety limit per map (default 900)
  HARNESS_RESULTS_DIR       where results go (default /results)
  HARNESS_TRACKS_DIR        track data + interpolate_track.py (default /competition/tracks)
  HARNESS_AGENT_FILE        the participant's file (default /agent/agent.py)
"""

import csv
import importlib.util
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from enum import IntEnum, auto
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MAPS = json.loads(os.environ['COMPETITION_MAPS'])
PARTICIPANT = os.environ['PARTICIPANT']
RUN_UUID = os.environ['RUN_UUID']
REPOSITORY = os.environ.get('REPOSITORY', '')
COMMIT_SHA = os.environ.get('COMMIT_SHA', '')

PORT = 8775
STEP_MS = 50                        # game time between decisions (TMRC's TM_STEP_MS default)
RACE_TIME_LIMIT_MS = 5 * 60 * 1000  # per map, in game time
RAY_ANGLES = [-90, -45, -10, 0, 10, 45, 90]  # TMRC track_features.RAY_ANGLES
RAY_MAX_DIST_M = 500.0
GAME_SPEED = float(os.environ.get('HARNESS_GAME_SPEED', '5'))
MAP_WALL_LIMIT_S = float(os.environ.get('HARNESS_MAP_WALL_LIMIT_S', '900'))
MAP_LOAD_LIMIT_S = 90               # real time for a map to load and its countdown to end
GAME_START_WAIT_S = 25              # real time for the game and TMInterface to start
CONNECT_ATTEMPTS = 12
PLUGIN_TIMEOUT_MS = 120_000         # plugin drops us if a step isn't answered within this

AGENT_FILE = Path(os.environ.get('HARNESS_AGENT_FILE', '/agent/agent.py'))
RESULTS_DIR = Path(os.environ.get('HARNESS_RESULTS_DIR', '/results'))
TRACKS_DIR = Path(os.environ.get('HARNESS_TRACKS_DIR', '/competition/tracks'))
MAPS_DIR = Path('/competition/maps')

TMNF_DIR = Path(os.environ.get('TMNF_DIR', '/home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever'))
CAMPAIGN_DIR = TMNF_DIR / 'GameData' / 'Tracks' / 'Campaigns'
WINE_USER_DIR = Path('/home/wineuser/.wine/drive_c/users/wineuser')
# `map <file>` resolves against the profile's Tracks/Challenges. Which
# Documents folder that is differs between images, so stage into each one.
CHALLENGE_DIRS = [WINE_USER_DIR / docs / 'TmForever' / 'Tracks' / 'Challenges'
                  for docs in ('Documents', 'My Documents')]

INPUT_KEYS = ('left', 'right', 'accelerate', 'brake')
TMI_KEYS = {'left': 'left', 'right': 'right', 'accelerate': 'up', 'brake': 'down'}
ERROR_INPUTS = {'left': False, 'right': False, 'accelerate': True, 'brake': False}
NO_INPUTS = {key: False for key in INPUT_KEYS}


# ---------------------------------------------------------------------------
# TMInterface protocol (must stay in sync with Python_Link.as / tminterface2.py)
# ---------------------------------------------------------------------------
class MessageType(IntEnum):
    SC_RUN_STEP_SYNC = auto()
    SC_CHECKPOINT_COUNT_CHANGED_SYNC = auto()
    SC_LAP_COUNT_CHANGED_SYNC = auto()
    SC_REQUESTED_FRAME_SYNC = auto()
    SC_ON_CONNECT_SYNC = auto()
    C_SET_SPEED = auto()
    C_REWIND_TO_STATE = auto()
    C_REWIND_TO_CURRENT_STATE = auto()
    C_GET_SIMULATION_STATE = auto()
    C_SET_INPUT_STATE = auto()
    C_GIVE_UP = auto()
    C_PREVENT_SIMULATION_FINISH = auto()
    C_SHUTDOWN = auto()
    C_EXECUTE_COMMAND = auto()
    C_SET_TIMEOUT = auto()
    C_RACE_FINISHED = auto()
    C_REQUEST_FRAME = auto()
    C_RESET_CAMERA = auto()
    C_SET_ON_STEP_PERIOD = auto()
    C_UNREQUEST_FRAME = auto()
    C_TOGGLE_INTERFACE = auto()
    C_IS_IN_MENUS = auto()
    C_GET_INPUTS = auto()


class Game:
    """One TMInterface connection, answering the plugin's sync messages."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.checkpoint = (0, 0)
        self.finish_time_ms = None

    def send_int(self, value):
        self.sock.sendall(struct.pack('i', int(value)))

    def recv_int(self):
        data = self.sock.recv(4, socket.MSG_WAITALL)
        if len(data) < 4:
            raise ConnectionError('TMInterface closed the connection')
        return struct.unpack('i', data)[0]

    def execute_command(self, command: str):
        data = command.encode()
        self.sock.sendall(struct.pack('ii', int(MessageType.C_EXECUTE_COMMAND), len(data)) + data)

    def set_speed(self, speed: float):
        self.sock.sendall(struct.pack('if', int(MessageType.C_SET_SPEED), speed))

    def set_on_step_period(self, period_ms: int):
        self.sock.sendall(struct.pack('ii', int(MessageType.C_SET_ON_STEP_PERIOD), period_ms))

    def set_timeout(self, timeout_ms: int):
        self.sock.sendall(struct.pack('iI', int(MessageType.C_SET_TIMEOUT), timeout_ms))

    def set_inputs(self, inputs: dict):
        self.sock.sendall(struct.pack('iBBBB', int(MessageType.C_SET_INPUT_STATE),
                                      *(int(inputs[key]) for key in INPUT_KEYS)))

    def get_state(self):
        from tminterface.structs import CheckpointData, SimStateData
        self.send_int(MessageType.C_GET_SIMULATION_STATE)
        length = self.recv_int()
        state = SimStateData(self.sock.recv(length, socket.MSG_WAITALL))
        state.cp_data.resize(CheckpointData.cp_states_field, state.cp_data.cp_states_length)
        state.cp_data.resize(CheckpointData.cp_times_field, state.cp_data.cp_times_length)
        return state

    def ack_step(self):
        self.send_int(MessageType.SC_RUN_STEP_SYNC)

    def wait_step(self):
        """Answer everything up to the next step. Returns (race_time_ms, sim state)."""
        while True:
            msg = self.recv_int()
            if msg == MessageType.SC_RUN_STEP_SYNC:
                race_time = self.recv_int()
                return race_time, self.get_state()
            if msg in (MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC, MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                current, target = self.recv_int(), self.recv_int()
                if msg == MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
                    self.checkpoint = (current, target)
                    if current == target and target > 0 and self.finish_time_ms is None:
                        # Before acking: otherwise the game ends the race, stops
                        # stepping, and the next map can't be loaded from here.
                        self.send_int(MessageType.C_PREVENT_SIMULATION_FINISH)
                        self.finish_time_ms = self.get_state().race_time
            self.send_int(msg)


# ---------------------------------------------------------------------------
# Track data: wall rays and distance along the centre line (TMRC's numbers)
# ---------------------------------------------------------------------------
class Track:
    """Rays from TMRC's TrackInterpolator, progress from track_features' maths.
    Without data for a map, every ray is (None, None) and distance is unknown."""

    def __init__(self, map_name: str):
        self.interp = None
        self.length_m = None
        csv_path = TRACKS_DIR / map_name / 'points-offset.csv'
        if not csv_path.exists():
            print(f'  No track data for {map_name}: rays are (None, None)')
            return
        if str(TRACKS_DIR) not in sys.path:
            sys.path.insert(0, str(TRACKS_DIR))
        from interpolate_track import TrackInterpolator, Vec3
        self.vec3 = Vec3
        self.interp = TrackInterpolator(csv_path=csv_path)
        with open(csv_path, newline='') as f:
            mids = np.array([[float(r['x']), float(r['z'])] for r in csv.DictReader(f) if r['type'] == 'mid'])
        seg = np.diff(mids, axis=0)
        self.seg_starts = mids[:-1]
        self.seg_vectors = seg
        self.seg_lengths = np.sqrt(np.sum(seg ** 2, axis=1))
        self.seg_lengths_sq = np.maximum(self.seg_lengths ** 2, 1e-12)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_lengths)])
        self.length_m = float(self.cum[-1])

    def rays(self, position, direction):
        if self.interp is None:
            return [(None, None)] * len(RAY_ANGLES)
        return self.interp.cast_rays(self.vec3(*position), self.vec3(*direction), RAY_ANGLES, RAY_MAX_DIST_M)

    def progress(self, position):
        if self.interp is None:
            return None
        to_car = np.array([position[0], position[2]]) - self.seg_starts
        t = np.clip(np.sum(to_car * self.seg_vectors, axis=1) / self.seg_lengths_sq, 0.0, 1.0)
        offsets = to_car - t[:, None] * self.seg_vectors
        i = int(np.argmin(np.sum(offsets ** 2, axis=1)))
        return float(self.cum[i] + t[i] * self.seg_lengths[i])

    def delta(self, prev, curr):
        d = curr - prev
        if d > self.length_m / 2:
            d -= self.length_m
        elif d < -self.length_m / 2:
            d += self.length_m
        return d


# ---------------------------------------------------------------------------
# Participant code
# ---------------------------------------------------------------------------
def load_agent(map_name: str):
    """Import agent.py fresh, as TMRC's AgentFilePolicy does, with
    TM_MAP_NAME set so map-dependent helpers (track_features) load this map."""
    os.environ['TM_MAP_NAME'] = map_name
    os.environ['TRACKS_DIR'] = str(TRACKS_DIR)
    agent_dir = str(AGENT_FILE.parent)
    # Helper modules next to agent.py are re-imported too.
    for name in [n for n, m in list(sys.modules.items())
                 if (getattr(m, '__file__', None) or '').startswith(agent_dir + os.sep)]:
        del sys.modules[name]
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    spec = importlib.util.spec_from_file_location(f'participant_agent_{map_name.replace("-", "_")}', AGENT_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'decide', None)):
        raise AttributeError('agent.py has no decide(game_state, wall_distances) function')
    return module.decide


def clean_inputs(action) -> dict:
    if not isinstance(action, dict):
        raise TypeError(f'decide() returned {type(action).__name__}, expected a dict')
    return {key: bool(action.get(key, False)) for key in INPUT_KEYS}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def stage_maps():
    """Put each competition map where `map <file>` finds it."""
    for m in MAPS:
        src = MAPS_DIR / m['file']
        if not src.exists():
            found = list(CAMPAIGN_DIR.rglob(m['file'])) if CAMPAIGN_DIR.exists() else []
            src = found[0] if found else None
        if src is None:
            print(f"WARNING: {m['file']} not found in {MAPS_DIR} or the campaign folders")
            continue
        for challenges in CHALLENGE_DIRS:
            if challenges.parent.parent.exists():
                challenges.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, challenges / m['file'])
        print(f"Staged {m['file']}")


def start_xvfb():
    os.environ['DISPLAY'] = ':99'
    # The base image ships a stale /tmp/.X99-lock (from its build) naming
    # PID 17. If any process in this container has that PID, Xvfb thinks :99
    # is taken and exits. Nothing else uses :99 here, so clear it.
    for stale in (Path('/tmp/.X99-lock'), Path('/tmp/.X11-unix/X99')):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
    xvfb = subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x16', '-nolisten', 'tcp'],
                            stdout=subprocess.DEVNULL, stderr=open('/tmp/xvfb.log', 'w'))
    sock = Path('/tmp/.X11-unix/X99')
    for _ in range(300):
        if sock.exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f'Xvfb did not start (exit code {xvfb.poll()}): '
                           + Path('/tmp/xvfb.log').read_text()[-500:])
    time.sleep(1)


def start_tmnf():
    exe = TMNF_DIR / 'TMLoader.exe'
    if not exe.exists():
        exe = TMNF_DIR / 'TmForever.exe'
    log = open('/tmp/tmnf.log', 'w')
    # vglrun renders on the GPU, but fails without one: only use it when the
    # container was given an NVIDIA device.
    gpu = Path('/dev/nvidiactl').exists()
    cmd = ['vglrun', 'wine', str(exe)] if gpu and shutil.which('vglrun') else ['wine', str(exe)]
    print(f"Launching: {' '.join(cmd)}")

    def _run():
        try:
            subprocess.Popen(cmd, stdout=log, stderr=log, cwd=str(exe.parent)).wait()
        except Exception as e:
            print(f'TMNF launch error: {e}')

    threading.Thread(target=_run, daemon=True).start()


def connect() -> Game:
    # No port polling: the plugin treats every TCP connect as a client and
    # waits for its handshake, so a probe would cost a 2 s timeout.
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect(('127.0.0.1', PORT))
            break
        except OSError as e:
            print(f'Connect attempt {attempt} failed: {e}')
            time.sleep(10)
    else:
        raise ConnectionError(f'TMInterface never came up on port {PORT}')
    game = Game(sock)
    game.send_int(MessageType.SC_ON_CONNECT_SYNC)  # pre-answer the plugin's handshake
    game.set_speed(GAME_SPEED)
    game.set_on_step_period(STEP_MS)
    game.set_timeout(PLUGIN_TIMEOUT_MS)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, struct.pack('ll', MAP_LOAD_LIMIT_S, 0))
    return game


# ---------------------------------------------------------------------------
# Per-map run
# ---------------------------------------------------------------------------
def load_map(game: Game, map_file: str, first: bool):
    """Load a map and return the first step at race time 0, after the countdown.
    For every map but the first, the previous map's last step is still
    unanswered when this is called."""
    game.execute_command(f'map {map_file}')
    if not first:
        game.set_inputs(NO_INPUTS)
        game.ack_step()
    deadline = time.time() + MAP_LOAD_LIMIT_S

    def next_step():
        if time.time() > deadline:
            raise TimeoutError(f'{map_file} did not load within {MAP_LOAD_LIMIT_S} s')
        return game.wait_step()

    race_time, state = next_step()
    # The previous map keeps stepping until the new one loads: wait for the
    # new countdown (negative race time)...
    while race_time >= 0 and not (first and race_time == 0):
        game.ack_step()
        race_time, state = next_step()
    # ...then step through it, where inputs do nothing.
    while race_time < 0:
        game.ack_step()
        race_time, state = next_step()
    # The map's checkpoint count (finish included), so a run that passes none
    # still reports "0/N". The plugin's checkpoint messages correct it if not.
    game.checkpoint = (0, int(state.cp_data.cp_states_length))
    game.finish_time_ms = None
    return race_time, state


def tmi_script(steps: list, end_ms: int) -> str:
    """TMInterface input script (`<from>-<to> press <key>`), as TMRC's replays write."""
    lines = []
    for key in INPUT_KEYS:
        start = None
        for t, inputs in steps:
            if inputs[key] and start is None:
                start = t
            elif not inputs[key] and start is not None:
                lines.append((start, f'{start}-{t} press {TMI_KEYS[key]}'))
                start = None
        if start is not None and end_ms > start:
            lines.append((start, f'{start}-{end_ms} press {TMI_KEYS[key]}'))
    return '\n'.join(line for _, line in sorted(lines)) + '\n'


def run_map(game: Game, map_info: dict, first: bool) -> dict:
    map_id, map_file = map_info['id'], map_info['file']
    map_name = map_file[:-len('.Challenge.Gbx')] if map_file.endswith('.Challenge.Gbx') else map_file
    print(f"\n=== {map_info['name']} ({map_file}) ===")

    result = {
        'id': map_id, 'name': map_info['name'], 'map_file': map_file,
        'finished': False, 'race_time_ms': None, 'end_reason': None,
        'checkpoints_passed': 0, 'total_checkpoints': 0, 'final_checkpoint_time_ms': None,
        'distance_m': None, 'track_length_m': None, 'track_pct': None,
        'steps': 0, 'decide_errors': 0, 'first_error': None,
        'decide_ms_mean': None, 'decide_ms_max': None, 'replay_file': None,
    }

    track = Track(map_name)
    if track.length_m:
        result['track_length_m'] = round(track.length_m, 1)
    try:
        decide = load_agent(map_name)
    except Exception:
        # Can't even import: every step counts as an error, so the car accelerates.
        decide = None
        result['first_error'] = 'loading agent.py: ' + traceback.format_exc(limit=3)
        print(f"  {result['first_error']}")

    race_time, state = load_map(game, map_file, first)
    prev_progress = track.progress(state.position)
    distance = 0.0 if prev_progress is not None else None
    steps, decide_ms = [], []
    last_cp = 0
    wall_deadline = time.time() + MAP_WALL_LIMIT_S

    while True:
        current_cp, target_cp = game.checkpoint
        if current_cp != last_cp:
            last_cp = current_cp
            result['final_checkpoint_time_ms'] = race_time
            print(f'  CP {current_cp}/{target_cp} at {race_time} ms')
        if game.finish_time_ms is not None:
            result['end_reason'] = 'finished'
            break
        if race_time >= RACE_TIME_LIMIT_MS:
            result['end_reason'] = 'time_limit'
            break
        if time.time() > wall_deadline:
            result['end_reason'] = 'wall_clock_limit'
            break

        rot = state.rotation_matrix
        game_state = {
            'position': tuple(float(v) for v in state.position),
            'velocity': tuple(float(v) for v in state.velocity),
            # Third column of the rotation matrix: the car's forward axis.
            'direction': (float(rot[0][2]), float(rot[1][2]), float(rot[2][2])),
        }
        wall_distances = track.rays(game_state['position'], game_state['direction'])
        started = time.perf_counter()
        try:
            if decide is None:
                raise RuntimeError('agent.py did not load')
            inputs = clean_inputs(decide(game_state, wall_distances))
        except Exception:
            inputs = dict(ERROR_INPUTS)
            result['decide_errors'] += 1
            if result['first_error'] is None:
                result['first_error'] = f'at {race_time} ms: ' + traceback.format_exc(limit=3)
                print(f"  decide() error (accelerating instead): {result['first_error']}")
        decide_ms.append((time.perf_counter() - started) * 1000)
        steps.append((race_time, inputs))

        game.set_inputs(inputs)
        game.ack_step()
        race_time, state = game.wait_step()
        if distance is not None:
            curr = track.progress(state.position)
            distance += track.delta(prev_progress, curr)
            prev_progress = curr

    current_cp, target_cp = game.checkpoint
    result['checkpoints_passed'] = current_cp
    result['total_checkpoints'] = target_cp
    if game.finish_time_ms is not None:
        result['finished'] = True
        result['race_time_ms'] = int(game.finish_time_ms)
        result['final_checkpoint_time_ms'] = int(game.finish_time_ms)
        print(f"  FINISHED: {result['race_time_ms']} ms")
    else:
        print(f"  Not finished ({result['end_reason']}) at {race_time} ms")
    if distance is not None:
        result['distance_m'] = round(distance, 1)
        result['track_pct'] = 100.0 if result['finished'] else round(100 * distance / track.length_m, 1)
    result['steps'] = len(steps)
    if decide_ms:
        result['decide_ms_mean'] = round(sum(decide_ms) / len(decide_ms), 3)
        result['decide_ms_max'] = round(max(decide_ms), 3)

    replay_name = f'{RUN_UUID}-{map_id}.inputs.txt'
    (RESULTS_DIR / replay_name).write_text(tmi_script(steps, result['race_time_ms'] or race_time))
    result['replay_file'] = replay_name
    print(f"  {result['steps']} decisions, {result['decide_errors']} errors, "
          f"distance {result['distance_m']} m, replay -> {replay_name}")
    return result


def failed_map(map_info: dict, reason: str) -> dict:
    return {'id': map_info['id'], 'name': map_info['name'], 'map_file': map_info['file'],
            'finished': False, 'race_time_ms': None, 'end_reason': reason,
            'checkpoints_passed': 0, 'total_checkpoints': 0, 'distance_m': None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('=== TMRC Competition Harness ===')
    print(f'UUID:        {RUN_UUID}')
    print(f'Participant: {PARTICIPANT}')
    print(f'Repository:  {REPOSITORY}')
    print(f'Commit:      {COMMIT_SHA}')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stage_maps()
    start_xvfb()
    start_tmnf()
    print(f'Waiting {GAME_START_WAIT_S} s for TMInterface...')
    time.sleep(GAME_START_WAIT_S)
    game = connect()
    print('Connected')

    map_results = []
    for i, map_info in enumerate(MAPS):
        try:
            map_results.append(run_map(game, map_info, first=(i == 0)))
        except Exception as e:
            # The game is in an unknown state: record this map as failed and
            # stop, rather than score the rest on a broken connection.
            print(f'  FAILED: {e}')
            traceback.print_exc()
            map_results.append(failed_map(map_info, f'harness_error: {e}'))
            map_results.extend(failed_map(rest, 'not_run') for rest in MAPS[i + 1:])
            break

    try:
        game.send_int(MessageType.C_SHUTDOWN)
        game.sock.close()
    except OSError:
        pass

    output = {
        'uuid': RUN_UUID,
        'participant': PARTICIPANT,
        'repository': REPOSITORY,
        'commit': COMMIT_SHA,
        'submitted_at': datetime.now(timezone.utc).isoformat(),
        'rules': {'step_ms': STEP_MS, 'race_time_limit_ms': RACE_TIME_LIMIT_MS,
                  'ray_angles': RAY_ANGLES, 'game_speed': GAME_SPEED},
        'maps': map_results,
    }
    out_path = RESULTS_DIR / f'{RUN_UUID}.json'
    out_path.write_text(json.dumps(output, indent=2))
    print(f'\nResults -> {out_path}')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
