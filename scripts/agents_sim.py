#!/usr/bin/env python3
"""Week 7 - the "live and die" forecaster agents (small version).

A population of simple reorder agents shares ONE shop's demand stream (the
contract sales CSV bucketed to weeks, same as backtest.py). Each agent
carries a virtual wallet scored, per the plan, on avoided stockouts AND
overstock:

    per week: wallet += sales revenue
                   - order cost
                   - stockout penalty (each lost unit costs)
                   - holding cost on leftover stock
                   - fixed weekly upkeep (an agent must earn its keep)

Agents start with house goods (AGENT_INITIAL_STOCK units per item) so the
cold start is fair: week-0 outcomes depend on the genome, not on the absence
of history.

All money is SIMULATION credits (unit_cost=1, unit_price=2) - not real
pricing, not a business claim.

Lifecycle (plan wording made precise):
  - DEATH at wallet <= 0: agent removed, event logged with lifespan/stats.
  - SPAWN: a living agent with wallet >= spawn_cost pays spawn_cost into a
    child whose genome (ma_k, safety) is mutated - the money comes from the
    parent. (A zero-balance estate would fund nothing, so parents fund
    children while alive; documented in docs/w7-agents-notes.md.)
  - LINEAGE_EXTINCT when the last member of a lineage dies; IMMIGRANT
    agents refill below pop_min (house money) so the watch never freezes.
  - lineage = founder id; generation increments each spawn.

Genome = (ma_k, safety): order up to ceil(mean(last ma_k weeks) * (1+safety))
per item, capped at max_order units - the same reorder math as reorder.py.

Dashboard: --html writes ONE self-contained HTML file that replays the event
stream (watch agents die and spawn, wallet bars, lineage chart) - dark
guard-style cards + event stream, patterns from stock-agent, no code copied.

Exit codes: 0 = ran, 1 = input/usage error.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import backtest as bt
except ImportError:                      # run as scripts/agents_sim.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import backtest as bt

# -- simulation economy (credits, not money) ------------------------------
INITIAL_WEALTH = 100.0
SPAWN_COST = 40.0
UNIT_COST = 1.0
UNIT_PRICE = 1.2           # slim margin: only efficient genomes stay solvent
STOCKOUT_PENALTY = 2.0     # per lost unit: missed margin + goodwill
HOLDING_COST = 0.35        # per leftover unit per week (overstock hurts)
UPKEEP = 10.0              # fixed cost per agent per week (must earn it)
AGENT_INITIAL_STOCK = 15.0  # house goods per item: fair cold start (no
                            # genome-independent week-0 wipeout)
MAX_ORDER = 40             # units per item per week (guard)
POP_MIN = 6
MAX_POP = 24
K_RANGE = (1, 12)
SAFETY_RANGE = (0.0, 0.6)


@dataclass
class Agent:
    id: str
    lineage: str
    gen: int
    parent: Optional[str]
    ma_k: int
    safety: float
    wallet: float
    born_week: int
    stock: Dict[str, float] = field(default_factory=dict)
    lost_units: float = 0.0
    leftover_unit_weeks: float = 0.0
    stockout_item_weeks: int = 0
    dead_week: Optional[int] = None

    @property
    def alive(self) -> bool:
        return self.dead_week is None


def mutate(ma_k: int, safety: float, rng: random.Random) -> Tuple[int, float]:
    """Random-candidate mutation that is GUARANTEED to change at least one
    gene (retry loop with a bounded fallback), so children never clone)."""
    for _ in range(20):
        k = min(K_RANGE[1], max(K_RANGE[0],
                ma_k + rng.choice((-2, -1, 1, 2))))
        s = min(SAFETY_RANGE[1], max(SAFETY_RANGE[0],
                round(safety + rng.uniform(-0.1, 0.1), 2)))
        if (k, s) != (ma_k, safety):
            return k, s
    k = K_RANGE[0] if ma_k == K_RANGE[0] else ma_k - 1
    return k, safety


def forecast(hist: Sequence[float], ma_k: int) -> float:
    """MA(k) over past weeks only (hist never includes week t - tested)."""
    if not hist:
        return 0.0
    window = hist[-ma_k:]
    return sum(window) / len(window)


def step_agent(a: Agent, demand: Dict[str, float],
               hists: Dict[str, List[float]]) -> None:
    """Advance one agent through one week. Order arrives before this week's
    demand (same assumption as backtest.run_stock); upkeep is charged first."""
    revenue = cost = stockout = holding = 0.0
    for item, d in demand.items():
        fc = forecast(hists.get(item, []), a.ma_k)
        target = math.ceil(fc * (1.0 + a.safety))
        on_hand = a.stock.get(item, 0.0)
        order = min(max(0, target - on_hand), MAX_ORDER)
        cost += order * UNIT_COST
        on_hand += order
        sold = min(on_hand, d)
        revenue += sold * UNIT_PRICE
        lost = d - sold
        if lost > 0:
            stockout += lost * STOCKOUT_PENALTY
            a.lost_units += lost
            a.stockout_item_weeks += 1
        on_hand -= sold
        a.stock[item] = on_hand
        holding += on_hand * HOLDING_COST
        a.leftover_unit_weeks += on_hand
    a.wallet += revenue - cost - stockout - holding - UPKEEP


def run_sim(weeks, series: Dict[str, List[Optional[float]]],
            seed: int = 7, initial_wealth: float = INITIAL_WEALTH,
            spawn_cost: float = SPAWN_COST, pop_min: int = POP_MIN,
            max_pop: int = MAX_POP) -> Dict:
    """Run the full simulation. Returns structured events + roster + summary
    (no file IO - tests call this directly)."""
    rng = random.Random(seed)
    items = sorted(series)
    n_weeks = len(weeks)
    roster: List[Agent] = []
    agents: List[Agent] = []
    events: List[Dict] = []
    next_id = 0

    def new_agent(genome, wallet, gen, lineage, parent, born) -> Agent:
        nonlocal next_id
        aid = f"A{next_id}"
        next_id += 1
        ag = Agent(id=aid, lineage=lineage, gen=gen, parent=parent,
                   ma_k=genome[0], safety=genome[1], wallet=wallet,
                   born_week=born,
                   stock={it: AGENT_INITIAL_STOCK for it in items})
        roster.append(ag)
        return ag

    # founders
    for _ in range(pop_min):
        k = rng.randint(*K_RANGE)
        s = round(rng.uniform(*SAFETY_RANGE), 2)
        ag = new_agent((k, s), initial_wealth, 0, f"L{next_id}", None, -1)
        agents.append(ag)
    events.append({"week": -1, "kind": "INIT",
                   "agents": [{"id": a.id, "lineage": a.lineage, "gen": a.gen,
                               "k": a.ma_k, "safety": a.safety,
                               "wallet": a.wallet} for a in agents]})

    lineages_dead: set = set()

    for t in range(n_weeks):
        demand = {it: float(series[it][t] or 0.0) for it in items}
        hists = {it: [v for v in series[it][:t] if v is not None]
                 for it in items}
        for a in agents:
            step_agent(a, demand, hists)

        # deaths (wallet <= 0)
        for a in list(agents):
            if a.wallet <= 0:
                a.dead_week = t
                agents.remove(a)
                events.append({"week": t, "kind": "DEAD", "id": a.id,
                               "lineage": a.lineage, "gen": a.gen,
                               "lifespan": t - a.born_week,
                               "wallet": round(a.wallet, 2),
                               "lost_units": round(a.lost_units, 1),
                               "leftover": round(a.leftover_unit_weeks, 1)})
                if not any(x.lineage == a.lineage for x in agents) \
                        and a.lineage not in lineages_dead:
                    lineages_dead.add(a.lineage)
                    events.append({"week": t, "kind": "LINEAGE_EXTINCT",
                                   "lineage": a.lineage, "gen": a.gen})

        # spawns: money comes from the parent
        for a in sorted(agents, key=lambda x: x.id):
            if a.wallet >= spawn_cost and len(agents) < max_pop:
                k, s = mutate(a.ma_k, a.safety, rng)
                a.wallet -= spawn_cost
                child = new_agent((k, s), spawn_cost, a.gen + 1, a.lineage,
                                  a.id, t)
                agents.append(child)
                events.append({"week": t, "kind": "SPAWN", "id": child.id,
                               "parent": a.id, "lineage": child.lineage,
                               "gen": child.gen, "k": k, "safety": s,
                               "wallet": spawn_cost,
                               "parent_wallet": round(a.wallet, 2)})

        # immigrants refill (house money) so the watch never freezes
        if len(agents) < pop_min:
            for _ in range(pop_min - len(agents)):
                k = rng.randint(*K_RANGE)
                s = round(rng.uniform(*SAFETY_RANGE), 2)
                ag = new_agent((k, s), initial_wealth, 0,
                               f"L{next_id}", None, t)
                agents.append(ag)
                events.append({"week": t, "kind": "IMMIGRANT", "id": ag.id,
                               "lineage": ag.lineage, "k": k, "safety": s,
                               "wallet": initial_wealth})

        events.append({"week": t, "kind": "WEEK", "pop": len(agents),
                       "wealth": round(sum(a.wallet for a in agents), 2),
                       "wallets": {a.id: round(a.wallet, 2)
                                   for a in agents},
                       "alive_ids": [a.id for a in agents]})

    summary = summarize(roster, n_weeks)
    return {"events": events, "roster": [asdict(a) for a in roster],
            "n_weeks": n_weeks, "weeks": [str(w) for w in weeks],
            "summary": summary,
            "params": {"seed": seed, "initial_wealth": initial_wealth,
                       "spawn_cost": spawn_cost, "pop_min": pop_min,
                       "max_pop": max_pop, "unit_cost": UNIT_COST,
                       "unit_price": UNIT_PRICE,
                       "stockout_penalty": STOCKOUT_PENALTY,
                       "holding_cost": HOLDING_COST, "upkeep": UPKEEP,
                       "agent_initial_stock": AGENT_INITIAL_STOCK,
                       "max_order": MAX_ORDER}}


def summarize(roster: Sequence[Agent], n_weeks: int) -> Dict:
    """Lineage-improvement stats: per generation + gen>=2 vs founders."""
    gens: Dict[int, List[Agent]] = {}
    for a in roster:
        gens.setdefault(a.gen, []).append(a)
    by_gen = []
    for g in sorted(gens):
        group = gens[g]
        end_w = [a.wallet for a in group]
        by_gen.append({
            "gen": g, "n": len(group),
            "mean_wallet_end": round(sum(end_w) / len(end_w), 2),
            "mean_lifespan": round(
                sum((a.dead_week if a.dead_week is not None else n_weeks)
                    - a.born_week for a in group) / len(group), 1),
        })
    founders = [a.wallet for a in gens.get(0, [])] or [0.0]
    later = [a.wallet for g, grp in gens.items() if g >= 2
             for a in grp] or [0.0]
    f_mean = sum(founders) / len(founders)
    l_mean = sum(later) / len(later)
    return {"by_gen": by_gen,
            "founders_mean_wallet": round(f_mean, 2),
            "descendants_mean_wallet": round(l_mean, 2),
            "descendants_better": l_mean > f_mean,
            "lineages_total": len({a.lineage for a in roster}),
            "agents_born": len(roster)}


# -- text log -------------------------------------------------------------
def render_log(result: Dict) -> str:
    out = []
    s = result["summary"]
    out.append(f"Week 7 agent sim - {result['n_weeks']} weeks, "
               f"seed {result['params']['seed']}")
    out.append(f"agents born: {s['agents_born']}  lineages: "
               f"{s['lineages_total']}")
    for e in result["events"]:
        w = e["week"]
        if e["kind"] == "DEAD":
            out.append(f"w{w:>3} DEAD     {e['id']} gen{e['gen']} "
                       f"{e['lineage']} life={e['lifespan']}w "
                       f"wallet={e['wallet']} lost={e['lost_units']} "
                       f"leftover={e['leftover']}")
        elif e["kind"] == "SPAWN":
            out.append(f"w{w:>3} SPAWN    {e['id']} <- {e['parent']} "
                       f"gen{e['gen']} k={e['k']} safety={e['safety']} "
                       f"(parent pays {result['params']['spawn_cost']})")
        elif e["kind"] == "IMMIGRANT":
            out.append(f"w{w:>3} ARRIVE   {e['id']} {e['lineage']} "
                       f"k={e['k']} safety={e['safety']}")
        elif e["kind"] == "LINEAGE_EXTINCT":
            out.append(f"w{w:>3} LINEAGE EXTINCT {e['lineage']}")
    out.append("")
    out.append("by generation (end of run):")
    out.append(f"{'gen':>4}{'n':>5}{'mean wallet':>14}{'mean lifespan':>15}")
    for row in s["by_gen"]:
        out.append(f"{row['gen']:>4}{row['n']:>5}"
                   f"{row['mean_wallet_end']:>14.2f}"
                   f"{row['mean_lifespan']:>13.1f}w")
    verdict = ("better" if s["descendants_better"] else "worse")
    out.append(f"lineage check: gen>=2 mean wallet "
               f"{s['descendants_mean_wallet']:.2f} vs founders "
               f"{s['founders_mean_wallet']:.2f} -> descendants {verdict}")
    return "\n".join(out)


# -- self-contained HTML dashboard ---------------------------------------
def render_html(result: Dict) -> str:
    data = json.dumps(result, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ShopBrain - live and die agents</title>
<style>
  :root {{ --bg:#0f1420; --panel:#161d2b; --line:#232c40; --text:#e6e9f2;
           --muted:#8b93a7; --accent:#6366f1; --green:#34d399;
           --red:#f87171; --amber:#fbbf24; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
          font:13px/1.45 Consolas, Menlo, monospace; }}
  header {{ padding:14px 18px; border-bottom:1px solid var(--line);
            display:flex; gap:26px; align-items:baseline; flex-wrap:wrap; }}
  h1 {{ font-size:15px; margin:0; letter-spacing:.5px; }}
  h1 span {{ color:var(--accent); }}
  .stat b {{ font-size:16px; }}
  .stat {{ color:var(--muted); }}
  .stat b {{ color:var(--text); display:block; }}
  .controls {{ margin-left:auto; display:flex; gap:8px; align-items:center; }}
  button {{ background:var(--panel); color:var(--text); border:1px solid var(--line);
            border-radius:6px; padding:5px 12px; font:inherit; cursor:pointer; }}
  button:hover {{ border-color:var(--accent); }}
  main {{ display:grid; grid-template-columns: 1fr 380px; gap:14px;
          padding:14px 18px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(168px,1fr));
            gap:10px; align-content:start; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
           padding:9px 11px; position:relative; overflow:hidden; }}
  .card.dead {{ opacity:.45; border-color:#3a2530; }}
  .card.dead::after {{ content:"DEAD"; position:absolute; top:7px; right:9px;
           color:var(--red); font-size:10px; letter-spacing:1px; }}
  .card.spawn {{ animation:pop .6s ease; }}
  @keyframes pop {{ 0%{{ transform:scale(.7); border-color:var(--green); }}
                    100%{{ transform:scale(1); }} }}
  .id {{ font-weight:bold; }} .gen {{ color:var(--muted); font-size:11px; }}
  .genome {{ color:var(--accent); font-size:12px; margin-top:3px; }}
  .wallet {{ margin-top:6px; font-size:12px; }}
  .bar {{ height:5px; background:#0b0f18; border-radius:3px; margin-top:4px; }}
  .bar i {{ display:block; height:100%; border-radius:3px;
            background:var(--green); transition:width .4s; }}
  .bar.neg i {{ background:var(--red); }}
  aside {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
           display:flex; flex-direction:column; min-height:420px; }}
  aside h2 {{ font-size:12px; color:var(--muted); margin:10px 12px;
              letter-spacing:1px; text-transform:uppercase; }}
  #stream {{ overflow-y:auto; padding:0 12px 12px; flex:1; font-size:12px; }}
  .ev {{ padding:2px 0; border-bottom:1px dashed #1c2436; white-space:nowrap; }}
  .ev .w {{ color:var(--muted); }}
  .ev.DEAD {{ color:var(--red); }} .ev.SPAWN {{ color:var(--green); }}
  .ev.IMMIGRANT {{ color:var(--amber); }} .ev.LINEAGE_EXTINCT {{ color:var(--red); }}
  footer {{ padding:10px 18px; border-top:1px solid var(--line);
            color:var(--muted); display:flex; gap:24px; align-items:center; }}
  canvas {{ background:var(--panel); border:1px solid var(--line);
            border-radius:8px; }}
  select {{ background:var(--panel); color:var(--text); border:1px solid var(--line);
            border-radius:6px; padding:4px; font:inherit; }}
</style>
</head>
<body>
<header>
  <h1>ShopBrain <span>/ live-and-die agents</span></h1>
  <div class="stat">week<b id="week">-</b></div>
  <div class="stat">population<b id="pop">-</b></div>
  <div class="stat">total wallet<b id="wealth">-</b></div>
  <div class="stat">born<b id="born">-</b></div>
  <div class="controls">
    <button id="play">Play</button>
    <select id="speed">
      <option value="700">slow</option>
      <option value="300" selected>normal</option>
      <option value="90">fast</option>
    </select>
    <button id="reset">Reset</button>
  </div>
</header>
<main>
  <div>
    <div class="cards" id="cards"></div>
    <footer style="border:0;padding:12px 0 0">
      <canvas id="chart" width="760" height="120"></canvas>
    </footer>
  </div>
  <aside><h2>event stream</h2><div id="stream"></div></aside>
</main>
<script>
const DATA = {data};
const $ = (id) => document.getElementById(id);
let idx = 0, timer = null, cards = {{}};

function card(a) {{
  const el = document.createElement('div');
  el.className = 'card';
  el.id = 'c-' + a.id;
  el.innerHTML = `<div class="id">${{a.id}} <span class="gen">gen${{a.gen}} ${{a.lineage}}</span></div>
    <div class="genome">MA(${{a.k}}) x ${{(1+a.safety).toFixed(2)}}</div>
    <div class="wallet" id="w-${{a.id}}">wallet ${{a.wallet.toFixed(1)}}</div>
    <div class="bar" id="b-${{a.id}}"><i style="width:0%"></i></div>`;
  return el;
}}
function setWallet(id, w) {{
  const b = $('b-' + id); if (!b) return;
  const pct = Math.max(0, Math.min(100, w / DATA.params.initial_wealth * 100));
  b.firstElementChild.style.width = pct + '%';
  b.classList.toggle('neg', w < 0);
  const t = $('w-' + id); if (t) t.textContent = 'wallet ' + w.toFixed(1);
}}
function stream(e, txt) {{
  const d = document.createElement('div');
  d.className = 'ev ' + e.kind;
  d.innerHTML = `<span class="w">w${{e.week}}</span> ${{txt}}`;
  const s = $('stream');
  s.prepend(d);
  while (s.children.length > 400) s.removeChild(s.lastChild);
}}
function apply(e) {{
  if (e.kind === 'INIT') {{
    e.agents.forEach(a => {{ $('cards').appendChild(card(a)); cards[a.id] = a; }});
    stream(e, `founders ${{e.agents.length}} agents`);
  }} else if (e.kind === 'SPAWN') {{
    const el = card(e); el.classList.add('spawn');
    $('cards').appendChild(el); cards[e.id] = e;
    setWallet(e.parent, e.parent_wallet); setWallet(e.id, e.wallet);
    stream(e, `SPAWN ${{e.id}} &lt;- ${{e.parent}} gen${{e.gen}} MA(${{e.k}}) x ${{(1+e.safety).toFixed(2)}} (parent pays ${{e.wallet}})`);
  }} else if (e.kind === 'DEAD') {{
    const el = $('c-' + e.id); if (el) el.classList.add('dead');
    setWallet(e.id, e.wallet);
    stream(e, `DEAD ${{e.id}} gen${{e.gen}} life=${{e.lifespan}}w lost=${{e.lost_units}} leftover=${{e.leftover}}`);
  }} else if (e.kind === 'IMMIGRANT') {{
    const el = card(e); $('cards').appendChild(el); cards[e.id] = e;
    stream(e, `ARRIVE ${{e.id}} ${{e.lineage}} MA(${{e.k}}) (house money)`);
  }} else if (e.kind === 'LINEAGE_EXTINCT') {{
    stream(e, `LINEAGE EXTINCT ${{e.lineage}}`);
  }} else if (e.kind === 'WEEK') {{
    $('week').textContent = e.week + 1;
    $('pop').textContent = e.pop;
    $('wealth').textContent = e.wealth.toFixed(0);
    $('born').textContent = DATA.summary.agents_born;
    e.alive_ids.forEach(id => {{ if (cards[id]) setWallet(id, cards[id].wallet); }});
    if (e.wallets) Object.entries(e.wallets).forEach(([id, w]) => setWallet(id, w));
    points.push([e.pop, e.wealth]); draw();
    if (e.week % 3 === 0) stream(e, `week ${{e.week + 1}} pop=${{e.pop}} wealth=${{e.wealth}}`);
  }}
}}
let points = [];
function draw() {{
  const c = $('chart'), ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (points.length < 2) return;
  const maxW = Math.max(...points.map(p => p[1]), 1);
  const maxP = Math.max(...points.map(p => p[0]), 1);
  const sx = c.width / (DATA.n_weeks - 1 || 1);
  const line = (get, max, color) => {{
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    points.forEach((p, i) => {{
      const x = i * sx, y = c.height - 6 - (get(p) / max) * (c.height - 14);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }});
    ctx.stroke();
  }};
  line(p => p[1], maxW, '#34d399');   // wealth
  line(p => p[0], maxP, '#6366f1');   // population
  ctx.fillStyle = '#8b93a7'; ctx.font = '10px Consolas';
  ctx.fillText('wallet (green) / population (indigo)', 8, 12);
}}
function step() {{
  if (idx >= DATA.events.length) {{ stop(); return; }}
  apply(DATA.events[idx++]);
}}
function stop() {{ clearInterval(timer); timer = null; $('play').textContent = 'Play'; }}
function start() {{
  if (timer) return stop();
  $('play').textContent = 'Pause';
  timer = setInterval(step, +$('speed').value);
}}
function reset() {{
  stop(); idx = 0; points = []; $('cards').innerHTML = ''; $('stream').innerHTML = '';
  $('week').textContent = '-'; $('pop').textContent = '-';
  $('wealth').textContent = '-'; $('born').textContent = '-';
  draw(); step();
}}
$('play').onclick = start;
$('reset').onclick = reset;
$('speed').onchange = () => {{ if (timer) {{ stop(); start(); }} }};
// show first frames immediately so the board is never empty
reset();
</script>
</body>
</html>
"""


# -- main -----------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sales", required=True, help="contract sales CSV")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pop", type=int, default=POP_MIN,
                    help=f"founding population (default {POP_MIN})")
    ap.add_argument("--max-pop", type=int, default=MAX_POP)
    ap.add_argument("--wealth", type=float, default=INITIAL_WEALTH)
    ap.add_argument("--spawn-cost", type=float, default=SPAWN_COST)
    ap.add_argument("--html", default=None,
                    help="write the replay dashboard to this path")
    ap.add_argument("--log", default=None,
                    help="write the text event log to this path")
    args = ap.parse_args(argv)

    try:
        if args.pop < 2:
            raise ValueError("--pop must be >= 2")
        if args.pop > args.max_pop:
            raise ValueError("--pop cannot exceed --max-pop")
        rows = bt.load_sales(args.sales)
        weeks, series = bt.weekly_series(rows)
        if len(weeks) < 8:
            raise ValueError(f"only {len(weeks)} weeks; need >= 8")
        result = run_sim(weeks, series, seed=args.seed,
                         initial_wealth=args.wealth,
                         spawn_cost=args.spawn_cost, pop_min=args.pop,
                         max_pop=args.max_pop)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    text = render_log(result)
    print(text)
    if args.log:
        Path(args.log).write_text(text + "\n", encoding="utf-8")
    if args.html:
        Path(args.html).write_text(render_html(result), encoding="utf-8")
        print(f"\ndashboard written: {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
