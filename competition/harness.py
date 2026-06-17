#!/usr/bin/env python3
"""
TMRC Competition Harness

Runs inside the participant's Docker image via:
  docker run --entrypoint python3 <image> /competition/harness.py

Replaces entrypoint.sh entirely: starts Xvfb, launches the game,
connects to TMInterface, and drives the race loop on each competition map.
Calls decide() from the participant's agent.py each simulation step.

Environment variables (set by evaluate.yml):
  COMPETITION_MAPS   JSON array: [{id, name, file}, ...]
  PARTICIPANT        GitHub username
  RUN_UUID           UUID for this evaluation run
  REPOSITORY         GitHub repo (org/submission-username)
  COMMIT_SHA         Git commit that triggered this run
"""

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


def sock_get_sim_state(sock):
    """Fetch position + velocity from the running simulation."""
    import numpy as np
    from tminterface.structs import CheckpointData, SimStateData
    sock.sendall(struct.pack("i", int(MessageType.C_GET_SIMULATION_STATE)))
    length = recv_int32(sock)
    raw = sock.recv(length, socket.MSG_WAITALL)
    state = SimStateData(raw)
    state.cp_data.resize(CheckpointData.cp_states_field, state.cp_data.cp_states_length)
    state.cp_data.resize(CheckpointData.cp_times_field, state.cp_data.cp_times_length)
    return {
        "position": (float(state.position[0]), float(state.position[1]), float(state.position[2])),
        "velocity": (float(state.velocity[0]), float(state.velocity[1]), float(state.velocity[2])),
    }


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
def run_map(sock, map_info: dict, decide_fn) -> dict:
    map_id = map_info['id']
    map_file = map_info['file']

    print(f"\n=== {map_info['name']} ({map_file}) ===")

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
                state = sock_get_sim_state(sock)
                action = decide_fn(last_race_time, state)
            except Exception as e:
                print(f"  decide() error: {e} — defaulting to accelerate")
                action = {}

            sock_set_input(
                sock,
                left=bool(action.get("left", False)),
                right=bool(action.get("right", False)),
                accelerate=bool(action.get("accelerate", True)),
                brake=bool(action.get("brake", False)),
            )
            send_int32(sock, MessageType.SC_RUN_STEP_SYNC)

        elif msg_type == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
            current = recv_int32(sock)
            target = recv_int32(sock)
            checkpoints_passed = current
            total_checkpoints = target
            final_checkpoint_time_ms = last_race_time
            print(f"  CP {current}/{target} at {last_race_time} ms")
            send_int32(sock, MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC)

            if current == target:
                finished = True
                race_time_ms = last_race_time
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
    print("Connected")

    map_results = []
    for map_info in MAPS:
        result = run_map(sock, map_info, decide)
        map_results.append(result)

    send_int32(sock, MessageType.C_SHUTDOWN)
    sock.close()

    output = {
        "uuid": RUN_UUID,
        "participant": PARTICIPANT,
        "repository": REPOSITORY,
        "commit": COMMIT_SHA,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "maps": map_results,
    }
    out_path = RESULTS_DIR / f"{RUN_UUID}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults → {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
