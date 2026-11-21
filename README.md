# gta6-trophy-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xxglitch/gta6-trophy-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xxglitch/gta6-trophy-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates on
Exophase: it reads the house's recording of the trophy page and forecasts
each of the 55 trophies' earn rate at 2026-11-26 00:00 UTC as five quantiles
(5%, 25%, 50%, 75%, 95%).

The page Exophase shows is not a live count. It is a snapshot that only
moves when Exophase refreshes the game's counts, every few hours, and a
forecast of the rate at the target is really a forecast of the snapshot the
page will show then: the rates as of the last refresh before the target. So
this program works in counts, not in rates, and steps from refresh to
refresh: between two snapshots of the page it fits, per trophy,
`dE = a·dN + b·U·dt` — the share of the joining players who already hold the
trophy, plus the per-hour hazard at which a player who does not hold it earns
it — fitted by non-negative least squares over the distinct snapshots the
case shows, shrunk toward pooled fits and the launch-night priors the
competition publishes (a ~4-hour refresh gap, 638 → 3,868 tracked players on
the first refresh, joiners holding other trophies at ~6.7% of the tracked
rate). From the last snapshot it simulates forward, refresh by refresh, over
a fixed grid of scenarios (refresh gap, growth per refresh, joiner profile,
progression hazard each at three levels), with the hazard decaying as the
horizon stretches — the players still without a trophy are increasingly the
slow ones — and reports the quantiles of the simulated rates.

Deterministic (two runs on the same case write the same bytes), standard
library only, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` reads every `*.in` case in `MAKEFUN_INPUT_DIR` (default `/input`)
and writes one `<id>.out` per case to `MAKEFUN_OUTPUT_DIR` (default
`/output`).

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
