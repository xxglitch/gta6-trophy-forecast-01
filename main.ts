// GTA VI trophy rarity forecast: a refresh-step cohort forecaster.
//
// The page Exophase shows is not a live count. It is a snapshot that only
// moves when Exophase refreshes the game's counts, every few hours, and a
// forecast of the rate at the target is really a forecast of the snapshot
// the page will show then: the rates as of the last refresh before the
// target. So this program works in counts, not in rates, and steps from
// refresh to refresh:
//
//   N(t)  players tracked, E_i(t)  players holding trophy i, r_i = 100 E_i / N.
//
// Between two snapshots of the page the only thing that can change is a
// refresh, and a refresh does two things at once: new players join the
// tracked set (N jumps), and the players already tracked have had time to
// earn more trophies. Per refresh the program therefore fits, per trophy,
//   dE_i = a_i * dN  +  b_i * U_i * dt
// where a_i is the share of the joining players who already hold trophy i,
// b_i the per-hour hazard at which a player who does not hold it earns it,
// U_i the players not holding it, dt the hours since the last snapshot.
// a_i and b_i are fitted by non-negative least squares over the snapshots
// the case shows, shrunk toward a pooled fit, with recent snapshots weighted
// more (the joiners of launch night are not the joiners of day five).
//
// What the snapshots cannot show, the priors below stand in for. They come
// from the competition's own published description of the launch night, not
// from any hidden answer: the page sat unchanged from the first probe
// (2026-11-18T23:48Z) through the 04:00Z read and refreshed by 04:30Z, so a
// refresh gap of about four hours; that refresh took the tracked players
// from 638 to 3,868 and the trophies earned from 943 to 4,276, so the
// joining players held the first trophy (every one of them) and almost
// nothing else, a joiner factor of about 0.067 on the other trophies. Where
// the case shows real refreshes those priors are outweighed by the fit.
//
// From the last snapshot the program simulates forward, refresh by refresh,
// to the target, over a fixed grid of scenarios (refresh gap, growth per
// refresh, joiner profile, progression hazard each at three levels) and
// reports the quantiles of the simulated rates. No clock, no randomness, no
// network: two runs on the same case write the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Refresh gap prior, hours: the launch-night page refreshed within ~4.7 h. */
const PRIOR_GAP_H = 4;
/** Growth prior: the documented first refresh, 638 -> 3,868 players. */
const PRIOR_GROWTH = 3868 / 638;
/** Joiner-factor prior: joining players held other trophies at ~6.7% of the
 *  tracked players' rate (3,333 joiner trophies over 3,230 joiners, 3,230 of
 *  them the first trophy, leaving 103 across the other 54). */
const PRIOR_JOINER_K = 0.067;
/** Progression-hazard prior scale: a trophy's per-hour hazard on players not
 *  holding it is this times its current rate, so a trophy almost nobody holds
 *  is almost nobody's next trophy either. (A flat hazard would have the
 *  platinum, held by 4 players, gained by a dozen more in four hours.) */
const PRIOR_HAZARD_SCALE = 0.05;
/** Growth per refresh decays this fast toward its floor (launch surge fades). */
const GROWTH_DECAY = 0.7;
/** Floor on the growth excess per refresh once the surge has faded. */
const GROWTH_FLOOR_EXCESS = 0.01;
/** Recency half-life, hours, for weighting snapshots in the fit. */
const HALF_LIFE_H = 12;
/** Hazard half-life, hours, over the forecast: the players who have not
 *  earned a trophy yet are increasingly the slow ones, so the per-hour
 *  hazard at which they earn it falls as the horizon stretches. Without
 *  this the model reads the fast early rise as a rate that never slows and
 *  overshoots every trophy that is already near its level. */
const HAZARD_HALF_LIFE_H = 72;
/** Prior strength, in pseudo-snapshots, when shrinking a trophy's fit. */
const PRIOR_WEIGHT = 2;
/** Scenario grid multipliers: gap, growth excess, joiner profile, hazard. */
const GRID = {
  gap: [0.85, 1, 1.18],
  growth: [0.7, 1, 1.4],
  joiner: [0.6, 1, 1.6],
  hazard: [0.5, 1, 2],
};

interface Trophy {
  id: number;
  name: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Weighted median of values with weights. */
function weightedMedian(values: number[], weights: number[]): number {
  const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const total = weights.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (const i of order) {
    acc += weights[i];
    if (acc >= total / 2) return values[i];
  }
  return values[order[order.length - 1]];
}
/** Weighted mean of values with weights. */
function weightedMean(values: number[], weights: number[]): number {
  let s = 0;
  let w = 0;
  for (let i = 0; i < values.length; i++) {
    s += weights[i] * values[i];
    w += weights[i];
  }
  return w > 0 ? s / w : 0;
}
/** Interpolated quantile of a sorted array at level tau. */
function quantile(sorted: number[], tau: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const pos = tau * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/**
 * Non-negative least squares for two regressors: minimise
 * sum w_j (z_j - a x_j - b y_j)^2 over a, b >= 0. Closed form: try the
 * unconstrained solution, then each boundary, and keep the best feasible.
 */
function nnls2(
  xs: number[],
  ys: number[],
  zs: number[],
  ws: number[],
): [number, number] {
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  let sxz = 0;
  let syz = 0;
  for (let j = 0; j < xs.length; j++) {
    const w = ws[j];
    sxx += w * xs[j] * xs[j];
    sxy += w * xs[j] * ys[j];
    syy += w * ys[j] * ys[j];
    sxz += w * xs[j] * zs[j];
    syz += w * ys[j] * zs[j];
  }
  const sse = (a: number, b: number): number => {
    let s = 0;
    for (let j = 0; j < xs.length; j++) {
      const d = zs[j] - a * xs[j] - b * ys[j];
      s += ws[j] * d * d;
    }
    return s;
  };
  let best: [number, number] = [0, 0];
  let bestSse = sse(0, 0);
  const det = sxx * syy - sxy * sxy;
  if (det > 1e-12) {
    const a = (sxz * syy - syz * sxy) / det;
    const b = (syz * sxx - sxz * sxy) / det;
    if (a >= 0 && b >= 0 && sse(a, b) < bestSse) {
      best = [a, b];
      bestSse = sse(a, b);
    }
  }
  if (syy > 0) {
    const b = Math.max(0, syz / syy);
    if (sse(0, b) < bestSse) {
      best = [0, b];
      bestSse = sse(0, b);
    }
  }
  if (sxx > 0) {
    const a = Math.max(0, sxz / sxx);
    if (sse(a, 0) < bestSse) {
      best = [a, 0];
      bestSse = sse(a, 0);
    }
  }
  return best;
}

/** One snapshot of the page: a distinct read (N and rates both unchanged means stale). */
interface Snapshot {
  tMs: number;
  n: number;
  total: number;
  earners: Array<number | null>;
}

function forecast(c: Case): Array<{ id: number; q: number[] }> {
  const nT = c.trophies.length;
  const targetMs = Date.parse(c.target);
  if (c.history.length === 0) throw new Error(`${c.id} has no history`);

  // Collapse the history to distinct snapshots: consecutive reads with the
  // same N and the same rates are one stale page, not new information.
  const snaps: Snapshot[] = [];
  for (const h of c.history) {
    const earners = h.rates.map((r) => (r === null ? null : (r * h.players_tracked) / 100));
    const prev = snaps[snaps.length - 1];
    if (
      prev !== undefined &&
      prev.n === h.players_tracked &&
      prev.earners.length === earners.length &&
      prev.earners.every((e, i) => e === earners[i])
    ) {
      continue;
    }
    snaps.push({ tMs: Date.parse(h.at), n: h.players_tracked, total: h.trophies_earned, earners });
  }
  const last = snaps[snaps.length - 1];
  const lastRates = last.earners.map((e) => (e === null ? 0 : (e / last.n) * 100));
  const m = snaps.length - 1; // number of transitions observed

  // ---- Fit the refresh process from the transitions. ----
  // Recency weights: a transition's weight halves every HALF_LIFE_H hours.
  const wts: number[] = [];
  for (let j = 1; j <= m; j++) {
    wts.push(Math.pow(0.5, (last.tMs - snaps[j].tMs) / (HALF_LIFE_H * 3600000)));
  }
  const gaps: number[] = [];
  const logGrowth: number[] = [];
  for (let j = 1; j <= m; j++) {
    gaps.push((snaps[j].tMs - snaps[j - 1].tMs) / 3600000);
    logGrowth.push(Math.log(snaps[j].n / snaps[j - 1].n));
  }
  const gapH = m > 0 ? clamp(weightedMedian(gaps, wts), 1, 12) : PRIOR_GAP_H;
  const growth = m > 0 ? clamp(Math.exp(weightedMean(logGrowth, wts)), 1.001, 20) : PRIOR_GROWTH;

  // Pooled fit on the totals: dTotal = A * dN + B * (non-earners) * dt.
  // B is an average hazard; the per-trophy hazard prior scales it by the
  // trophy's rate (see PRIOR_HAZARD_SCALE), so rare trophies progress slowly.
  // Every transition is divided by the players tracked before it, so the
  // fit reads relative moves: in raw counts the launch surge's huge
  // transitions would drown the recent, smaller ones the future resembles.
  let poolA = 0;
  let poolB = 0;
  if (m > 0) {
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    for (let j = 1; j <= m; j++) {
      const prev = snaps[j - 1];
      const cur = snaps[j];
      const nonEarners = c.trophies.length * prev.n - prev.total;
      const scale = prev.n > 0 ? prev.n : 1;
      xs.push((cur.n - prev.n) / scale);
      ys.push((Math.max(0, nonEarners) / scale) * ((cur.tMs - prev.tMs) / 3600000));
      zs.push((cur.total - prev.total) / scale);
    }
    [poolA, poolB] = nnls2(xs, ys, zs, wts);
  }
  let hazardScale = PRIOR_HAZARD_SCALE;
  if (m > 0 && poolB > 0) {
    const sumU = last.earners.reduce((a, e) => a + Math.max(0, last.n - (e ?? 0)), 0);
    const sumRU = last.earners.reduce(
      (a, e, i) => a + (lastRates[i] / 100) * Math.max(0, last.n - (e ?? 0)),
      0,
    );
    if (sumRU > 0) hazardScale = (poolB * sumU) / sumRU;
  }

  // Joiner-profile prior: a_i = 1 - (1 - r_i/100)^K, with K chosen so the
  // joining players' trophies per head match the pooled fit (or the launch
  // night's 0.067 where there is no fit to match).
  const sumRates = lastRates.reduce((a, b) => a + b, 0);
  let joinerK = PRIOR_JOINER_K;
  if (m > 0 && sumRates > 0 && poolA > 0) {
    // Solve sum_i [1 - (1 - r_i/100)^K] = poolA by bisection on K in [0, 50].
    let lo = 0;
    let hi = 50;
    const totalAt = (k: number): number =>
      lastRates.reduce((a, r) => a + (1 - Math.pow(1 - r / 100, k)), 0);
    if (totalAt(hi) >= poolA) {
      for (let it = 0; it < 60; it++) {
        const mid = (lo + hi) / 2;
        if (totalAt(mid) < poolA) lo = mid;
        else hi = mid;
      }
      joinerK = (lo + hi) / 2;
    } else {
      joinerK = hi;
    }
  }
  const joinerPrior = (r: number): number =>
    r >= 99.99 ? 1 : 1 - Math.pow(1 - r / 100, joinerK);

  // Per-trophy fit, shrunk toward the pooled profile and hazard.
  const alpha: number[] = new Array(nT).fill(0);
  const beta: number[] = new Array(nT).fill(0);
  for (let i = 0; i < nT; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    const ws: number[] = [];
    for (let j = 1; j <= m; j++) {
      const prev = snaps[j - 1];
      const cur = snaps[j];
      if (prev.earners[i] === null || cur.earners[i] === null) continue;
      const scale = prev.n > 0 ? prev.n : 1;
      xs.push((cur.n - prev.n) / scale);
      ys.push((Math.max(0, prev.n - (prev.earners[i] as number)) / scale) * ((cur.tMs - prev.tMs) / 3600000));
      zs.push(((cur.earners[i] as number) - (prev.earners[i] as number)) / scale);
      ws.push(wts[j - 1]);
    }
    const wSum = ws.reduce((a, b) => a + b, 0);
    const hazardPrior = hazardScale * (lastRates[i] / 100);
    if (xs.length > 0 && wSum > 0) {
      const [aFit, bFit] = nnls2(xs, ys, zs, ws);
      alpha[i] = (wSum * aFit + PRIOR_WEIGHT * joinerPrior(lastRates[i])) / (wSum + PRIOR_WEIGHT);
      beta[i] = (wSum * bFit + PRIOR_WEIGHT * hazardPrior) / (wSum + PRIOR_WEIGHT);
    } else {
      alpha[i] = joinerPrior(lastRates[i]);
      beta[i] = hazardPrior;
    }
    alpha[i] = clamp(alpha[i], 0, 1);
    beta[i] = Math.max(0, beta[i]);
  }

  // ---- Simulate forward, refresh by refresh, over the scenario grid. ----
  // The answer is the page as of the last refresh at or before the target:
  // rates only move at simulated refreshes, never between them.
  const sims: number[][] = c.trophies.map(() => []);
  for (const gGap of GRID.gap) {
    for (const gGrowth of GRID.growth) {
      for (const gJoin of GRID.joiner) {
        for (const gHaz of GRID.hazard) {
          const stepH = gapH * gGap;
          let n = last.n;
          const earners = last.earners.map((e) => (e === null ? 0 : e));
          let t = last.tMs;
          let k = 0;
          while (t + stepH * 3600000 <= targetMs) {
            t += stepH * 3600000;
            k++;
            // The hazard falls as the horizon stretches: the players still
            // without a trophy are the slow ones.
            const hazardDecay = Math.pow(0.5, (t - last.tMs) / (HAZARD_HALF_LIFE_H * 3600000));
            // Growth per refresh, scaled by this scenario, decaying as the
            // launch surge fades toward a small steady inflow.
            const excess = Math.max(GROWTH_FLOOR_EXCESS, (growth - 1) * Math.pow(GROWTH_DECAY, k - 1));
            const gNscen = 1 + excess * gGrowth;
            const dN = n * (gNscen - 1);
            for (let i = 0; i < nT; i++) {
              const nonEarners = Math.max(0, n - earners[i]);
              // The joiner share scales on the odds: a trophy everybody holds
              // stays held by every joiner in every scenario, and a share of
              // 0.001 moves by the scenario factor, not by a power of itself
              // (a power would turn "joiners hold it 15x more" into "joiners
              // hold it as often as the tracked players do").
              let aScen = alpha[i];
              if (aScen > 0 && aScen < 1) {
                const odds = (aScen / (1 - aScen)) * gJoin;
                aScen = odds / (1 + odds);
              }
              earners[i] += aScen * dN + nonEarners * (1 - Math.exp(-beta[i] * gHaz * hazardDecay * stepH));
              earners[i] = Math.min(earners[i], n + dN);
            }
            n += dN;
          }
          for (let i = 0; i < nT; i++) {
            sims[i].push((earners[i] / n) * 100);
          }
        }
      }
    }
  }

  // ---- Quantiles of the simulated rates, at the case's levels. ----
  const rows = c.trophies.map((t, i) => {
    const sorted = sims[i].slice().sort((a, b) => a - b);
    let prev = -Infinity;
    const q = c.levels.map((tau) => {
      const v = round4(clamp(quantile(sorted, tau), 0.01, 99.99));
      const out = Math.max(v, prev);
      prev = out;
      return out;
    });
    return { id: t.id, q };
  });
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecast(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
