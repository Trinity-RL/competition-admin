# TMRC

## Join the Trinity-RL Trackmania Competition

1. Make sure you have a GitHub account.
2. [Open a new issue using the "Join the competition" template](https://github.com/Trinity-RL/competition-admin/issues/new?template=join-competition.yml).
3. Enter your Trinity Number as the Title after `[join]`
4. Enter your GitHub username (not your email) and submit the issue.
5. Within a minute or two, a bot will comment on your issue and close it once your access is ready. You'll get:
   - Read access to [`tmrc-demo`](https://github.com/Trinity-RL/tmrc-demo) — the framework, agent interface, and a working baseline agent to build from.
   - Your own private submission repo (`submission-<your-username>`) that only you and the organizers can see.
6. Check your GitHub notifications/email for two repository invitations and **accept both** — access isn't active until you do.
7. Clone your submission repo and start building your bot against the interface in `tmrc-demo`.

If something goes wrong, the bot will comment on your issue explaining the failure instead of closing it — an organizer will follow up.

## How submissions are run

`competition/harness.py` imports `decide()` from your `agent/agent.py` and calls it every
50 ms of game time on each map in `maps.yml`, with the same inputs as the
[TMRC framework](https://github.com/ZSH4RK/TMRC):

```python
def decide(game_state, wall_distances):  # -> {"left", "right", "accelerate", "brake"}
```

- Each map is one attempt, limited to 5 minutes of game time. Finishers are ranked by time,
  everyone else by distance along the track.
- `agent.py` is imported fresh for each map, with `TM_MAP_NAME` set to that map.
- If `decide()` raises an error or returns something unusable, the car accelerates for that
  step and the error is counted in your results.
- `wall_distances` comes from the [Tracks](https://github.com/Trinity-RL/Tracks) data for
  A01, A02 and A03. The copy used is in `tracks/`, and `tracks/SOURCE` names the Tracks
  commit it came from.
- Bots written for the original `decide(race_time_ms, state)` interface still run.

Full rules: TMRC's `docs/11-competition.md`.
