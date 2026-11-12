#!/usr/bin/env python3
"""
TMRC Competition Harness

Runs inside the participant's Docker image via:
  docker run --entrypoint python3 <image> /competition/harness.py

Replaces entrypoint.sh entirely: starts Xvfb, launches the game,
connects to TMInterface, and drives the race loop on each competition map.
Calls decide() from the participant's agent.py each simulation step.

Agent interface (the same as the TMRC framework's, github.com/ZSH4RK/TMRC):
  decide(game_state, wall_distances) -> {left, right, accelerate, brake}
    game_state:     {"position", "velocity", "direction"} (3-tuples)
    wall_distances: 7 (distance_m or None, "left"/"right"/None) pairs at
                    RAY_ANGLES, computed from Trinity-RL/Tracks data; all
                    (None, None) on maps without track data
  Agents written for the original decide(race_time_ms, state) interface are
  detected by their first parameter's name and still called that way.

Environment variables (set by evaluate.yml):
  COMPETITION_MAPS   JSON array: [{id, name, file}, ...]
  PARTICIPANT        GitHub username
  RUN_UUID           UUID for this evaluation run
  REPOSITORY         GitHub repo (org/submission-username)
  COMMIT_SHA         Git commit that triggered this run
  TM_STEP_MS         Game time between decide() calls (default 50, as in TMRC)
"""

import inspect
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from enum import IntEnum, auto
from pathlib import Path

sys.path.insert(0, '/agent')

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
MAPS = json.loads(os.environ['COMPETITION_MAPS'])
PARTICIPANT = os.environ['PARTICIPANT']
RUN_UUID = os.environ['RUN_UUID']
REPOSITORY = os.environ.get('REPOSITORY', '')
COMMIT_SHA = os.environ.get('COMMIT_SHA', '')

PORT = 8775
# Game time between decide() calls. Must match the TMRC framework's default,
# because agents trained there act at that rate and drive differently at others.
STEP_MS = int(os.environ.get('TM_STEP_MS', '50'))
# Same angles, order and maths as TMRC's track_features.RAY_ANGLES.
RAY_ANGLES = [-90, -45, -10, 0, 10, 45, 90]
# Trinity-RL/Tracks: mounted by evaluate.yml; images built from the current
# template also carry a copy at /opt/tracks.
TRACKS_DIRS = [Path('/competition/tracks'), Path('/opt/tracks')]
# decide() errors printed per map; all are counted in the results.
MAX_PRINTED_ERRORS = 5
MAP_TIMEOUT_SECONDS = 300  # 5 min per map
MAP_LOAD_WAIT = 6          # seconds to wait after execute_command('map ...')
POST_RUN_DRAIN = 3         # seconds to drain socket after each map

RESULTS_DIR = Path('/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TMNF_DIR = Path(os.environ.get('TMNF_DIR', '/home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever'))
WINE_DOCS = Path('/home/wineuser/.wine/drive_c/users/wineuser/My Documents/TrackMania')
TRACKS_DIR = WINE_DOCS / 'Tracks' / 'Challenges'
REPLAYS_DIR = WINE_DOCS / 'Replays'


# ---------------------------------------------------------------------------
# TMInterface message types (must stay in sync with tminterface2.py)
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


# ---------------------------------------------------------------------------
# Socket helpers
# ---------------------------------------------------------------------------
def recv_int32(sock):
    return struct.unpack("i", sock.recv(4, socket.MSG_WAITALL))[0]


def send_int32(sock, value):
    sock.sendall(struct.pack("i", int(value)))


def sock_execute_command(sock, command: str):
    data = command.encode()
    sock.sendall(struct.pack("ii", int(MessageType.C_EXECUTE_COMMAND), len(data)))
    sock.sendall(data)


def sock_set_input(sock, left=False, right=False, accelerate=False, brake=False):
    sock.sendall(struct.pack(
        "iBBBB",
        int(MessageType.C_SET_INPUT_STATE),
        int(left), int(right), int(accelerate), int(brake),
    ))


def sock_set_step_period(sock, period_ms: int):
    sock.sendall(struct.pack("ii", int(MessageType.C_SET_ON_STEP_PERIOD), int(period_ms)))


def sock_get_raw_state(sock):
    """Fetch the full simulation state."""
    import numpy as np
    from tminterface.structs import CheckpointData, SimStateData
    sock.sendall(struct.pack("i", int(MessageType.C_GET_SIMULATION_STATE)))
    length = recv_int32(sock)
    raw = sock.recv(length, socket.MSG_WAITALL)
    state = SimStateData(raw)
    state.cp_data.resize(CheckpointData.cp_states_field, state.cp_data.cp_states_length)
    state.cp_data.resize(CheckpointData.cp_times_field, state.cp_data.cp_times_length)
    return state


def sock_get_sim_state(sock):
    """Fetch position, velocity and facing direction, as the TMRC framework reads them."""
    state = sock_get_raw_state(sock)
    rot = state.rotation_matrix  # columns are the car's local axes; the third is forward
    return {
        "position": (float(state.position[0]), float(state.position[1]), float(state.position[2])),
        "velocity": (float(state.velocity[0]), float(state.velocity[1]), float(state.velocity[2])),
        "direction": (float(rot[0][2]), float(rot[1][2]), float(rot[2][2])),
    }


# ---------------------------------------------------------------------------
# Agent interface
# ---------------------------------------------------------------------------
class WallSensor:
    """Wall distances along RAY_ANGLES for one map, computed with Tracks'
    interpolate_track.TrackInterpolator exactly as the TMRC framework does."""

    def __init__(self, map_name: str):
        self.map_name = map_name
        self.track, self.Vec3, self.source = None, None, None
        for tracks_dir in TRACKS_DIRS:
            csv = tracks_dir / map_name / 'points-offset.csv'
            if not csv.exists():
                continue
            try:
                if str(tracks_dir) not in sys.path:
                    sys.path.insert(0, str(tracks_dir))
                from interpolate_track import TrackInterpolator, Vec3
                self.track, self.Vec3, self.source = TrackInterpolator(csv_path=csv), Vec3, str(csv)
                print(f"  Track data: {csv}")
            except Exception as e:
                print(f"  WARNING: could not load track data {csv}: {e}")
            break
        if self.track is None:
            print(f"  No track data for {map_name}: wall_distances will all be (None, None)")

    def __call__(self, game_state):
        if self.track is None:
            return [(None, None)] * len(RAY_ANGLES)
        return self.track.cast_rays(self.Vec3(*game_state["position"]),
                                    self.Vec3(*game_state["direction"]), RAY_ANGLES)


def make_driver(decide):
    """Wrap decide() as driver(race_time_ms, game_state, wall_distances) -> action.

    Returns (interface name, driver, default for a missing key). The original
    interface is recognised by its first parameter, race_time_ms."""
    try:
        params = list(inspect.signature(decide).parameters)
    except (TypeError, ValueError):
        params = []
    if params and params[0].startswith('race_time'):
        def legacy(race_time_ms, game_state, _walls):
            return decide(race_time_ms, {"position": game_state["position"], "velocity": game_state["velocity"]})
        # The original harness treated a missing "accelerate" as pressed.
        return 'legacy', legacy, True
    return 'tmrc', (lambda _race_time_ms, game_state, walls: decide(game_state, walls)), False


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def stage_maps():
    """Copy competition .gbx files into Tracks/Challenges/ so execute_command finds them."""
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    for m in MAPS:
        src = Path('/competition/maps') / m['file']
        dst = TRACKS_DIR / m['file']
        if src.exists():
            shutil.copy(src, dst)
            print(f"Staged {m['file']}")
        else:
            print(f"WARNING: {src} not found — map must already exist in game data")


def start_xvfb():
    os.environ['DISPLAY'] = ':99'
    subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x16', '-nolisten', 'tcp'])
    time.sleep(3)


def start_tmnf():
    exe = TMNF_DIR / 'TMLoader.exe'
    if not exe.exists():
        exe = TMNF_DIR / 'TmForever.exe'
    log = open('/tmp/tmnf.log', 'w')

    # Use vglrun for GPU-accelerated rendering if available, otherwise plain wine
    import shutil
    cmd = (['vglrun', 'wine', str(exe)] if shutil.which('vglrun')
           else ['wine', str(exe)])
    print(f"Launching: {' '.join(cmd)}")

    def _run():
        try:
            proc = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=str(exe.parent))
            proc.wait()
        except Exception as e:
            print(f"TMNF launch error: {e}")

    threading.Thread(target=_run, daemon=True).start()


def find_new_replay(before: set) -> Path | None:
    """Return the newest .Replay.Gbx that appeared since before was captured."""
    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    time.sleep(2)
    after = set(REPLAYS_DIR.glob('*.Replay.Gbx'))
    new = after - before
    return max(new, key=lambda p: p.stat().st_mtime) if new else None


def drain_socket(sock, duration: float):
    """Discard incoming messages for `duration` seconds to clear the socket after a map ends."""
    deadline = time.time() + duration
    sock.settimeout(0.3)
    while time.time() < deadline:
        try:
            msg_type = recv_int32(sock)
        except (socket.timeout, OSError):
            break
        # Consume any payload and echo back to unblock the plugin
        if msg_type == int(MessageType.SC_RUN_STEP_SYNC):
            try:
                recv_int32(sock)  # race_time
            except OSError:
                break
        elif msg_type == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
            try:
                recv_int32(sock)  # current
                recv_int32(sock)  # target
            except OSError:
                break
        elif msg_type == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
            try:
                recv_int32(sock)  # current
                recv_int32(sock)  # target
            except OSError:
                break
        try:
            send_int32(sock, msg_type)
        except OSError:
            break
    sock.settimeout(None)


# ---------------------------------------------------------------------------
# Per-map run
# ---------------------------------------------------------------------------
def run_map(sock, map_info: dict, driver, accelerate_default: bool) -> dict:
    map_id = map_info['id']
    map_file = map_info['file']

    print(f"\n=== {map_info['name']} ({map_file}) ===")
    sensor = WallSensor(map_info.get('track', map_file.replace('.Challenge.Gbx', '')))
    decide_errors = 0

    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    before_replays = set(REPLAYS_DIR.glob('*.Replay.Gbx'))

    sock_execute_command(sock, f'map {map_file}')
    time.sleep(MAP_LOAD_WAIT)

    last_race_time = 0
    checkpoints_passed = 0
    total_checkpoints = 0
    final_checkpoint_time_ms = None
    finished = False
    race_time_ms = None
    deadline = time.time() + MAP_TIMEOUT_SECONDS

    while True:
        if time.time() > deadline:
            print(f"  Timeout after {MAP_TIMEOUT_SECONDS}s")
            send_int32(sock, MessageType.C_GIVE_UP)
            break

        try:
            msg_type = recv_int32(sock)
        except Exception as e:
            print(f"  Socket error: {e}")
            break

        if msg_type == int(MessageType.SC_RUN_STEP_SYNC):
            last_race_time = recv_int32(sock)

            try:
                game_state = sock_get_sim_state(sock)
                action = driver(last_race_time, game_state, sensor(game_state))
                if not isinstance(action, dict):
                    raise TypeError(f"decide() returned {type(action).__name__}, expected dict")
                accelerate = action.get("accelerate", accelerate_default)
            except Exception as e:
                decide_errors += 1
                if decide_errors <= MAX_PRINTED_ERRORS:
                    print(f"  decide() error at {last_race_time} ms: {type(e).__name__}: {e} — accelerating this step")
                action, accelerate = {}, True

            sock_set_input(
                sock,
                left=bool(action.get("left", False)),
                right=bool(action.get("right", False)),
                accelerate=bool(accelerate),
                brake=bool(action.get("brake", False)),
            )
            send_int32(sock, MessageType.SC_RUN_STEP_SYNC)

        elif msg_type == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
            current = recv_int32(sock)
            target = recv_int32(sock)
            checkpoints_passed = current
            total_checkpoints = target
            # The exact time from the game: last_race_time is the latest decision
            # step, up to STEP_MS - 10 ms before the checkpoint was crossed.
            # Read before acking, while the game is paused on this message.
            try:
                cp_time_ms = int(sock_get_raw_state(sock).race_time)
            except Exception as e:
                print(f"  Could not read exact checkpoint time ({e}); using {last_race_time} ms")
                cp_time_ms = last_race_time
            final_checkpoint_time_ms = cp_time_ms
            print(f"  CP {current}/{target} at {cp_time_ms} ms")
            send_int32(sock, MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC)

            if current == target:
                finished = True
                race_time_ms = cp_time_ms
                print(f"  FINISHED: {race_time_ms} ms")
                # Give up triggers automatic replay save
                send_int32(sock, MessageType.C_GIVE_UP)
                break

        elif msg_type == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
            recv_int32(sock)  # current
            recv_int32(sock)  # target
            send_int32(sock, MessageType.SC_LAP_COUNT_CHANGED_SYNC)

        elif msg_type == int(MessageType.SC_ON_CONNECT_SYNC):
            send_int32(sock, MessageType.SC_ON_CONNECT_SYNC)

        else:
            send_int32(sock, msg_type)

    # Drain socket so the next map command isn't confused by leftover messages
    drain_socket(sock, POST_RUN_DRAIN)

    # Collect replay
    replay_src = find_new_replay(before_replays)
    replay_filename = None
    if replay_src:
        replay_filename = f"{RUN_UUID}-{map_id}.Replay.Gbx"
        shutil.copy(replay_src, RESULTS_DIR / replay_filename)
        print(f"  Replay saved → {replay_filename}")
    else:
        print(f"  No replay found for {map_id}")

    return {
        "id": map_id,
        "name": map_info['name'],
        "finished": finished,
        "race_time_ms": race_time_ms,
        "checkpoints_passed": checkpoints_passed,
        "total_checkpoints": total_checkpoints,
        "final_checkpoint_time_ms": final_checkpoint_time_ms,
        "replay_file": replay_filename,
        "track_data": sensor.source is not None,
        "decide_errors": decide_errors,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== TMRC Competition Harness ===")
    print(f"UUID:        {RUN_UUID}")
    print(f"Participant: {PARTICIPANT}")
    print(f"Repository:  {REPOSITORY}")
    print(f"Commit:      {COMMIT_SHA}")

    # Import participant's decision function
    try:
        from agent import decide
    except Exception as e:
        print(f"FATAL: could not import decide() from agent.py: {e}")
        sys.exit(1)
    interface, driver, accelerate_default = make_driver(decide)
    print(f"Agent interface: {interface}, decide() every {STEP_MS} ms")

    stage_maps()
    start_xvfb()
    start_tmnf()
    print("Waiting 20 s for TMInterface...")
    time.sleep(20)

    print("Connecting to TMInterface...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect(('127.0.0.1', PORT))

    # Pre-send SC_ON_CONNECT_SYNC — prevents the 2 s timeout in the plugin
    send_int32(sock, MessageType.SC_ON_CONNECT_SYNC)
    sock_set_step_period(sock, STEP_MS)
    print("Connected")

    map_results = []
    for map_info in MAPS:
        result = run_map(sock, map_info, driver, accelerate_default)
        map_results.append(result)

    send_int32(sock, MessageType.C_SHUTDOWN)
    sock.close()

    output = {
        "uuid": RUN_UUID,
        "participant": PARTICIPANT,
        "repository": REPOSITORY,
        "commit": COMMIT_SHA,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "interface": interface,
        "step_ms": STEP_MS,
        "maps": map_results,
    }
    out_path = RESULTS_DIR / f"{RUN_UUID}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults → {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
