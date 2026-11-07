# prop-lab

A research lab and paper trader for **NQ/MNQ futures** (with prop firm
evaluations as the goal) and **Solana memecoins**. One rule runs everything:

> **If it can't beat a coin flip, it doesn't trade.**

Nothing in this repo places real orders. Live trading is a later phase, and it
only gets built for a strategy that has passed every stage below, including
weeks of paper trading.

---

## The funnel

| Step | What it does | Command |
|---|---|---|
| 1. Data | Buy/download 1-minute bars | `scripts/download_databento.py`, `scripts/download_solana_history.py` |
| 2. Honest backtest | Costs + slippage on every fill, next-bar fills, stop-first, no lookahead (checked automatically) | built into the labs |
| 3. Unseen years | Build period vs never-seen period side by side, plus a locked holdout | `--is-end`, `--holdout-months` |
| 4. Cost stress | Re-runs everything at 2x costs | automatic |
| 5. Coin flip | Same trades/stops/holding times, random entries, 500 times | `--flips 500` |
| 6. Research log | Every test logged *before* it runs, tamper-evident, counts your variations | `research_log/log.jsonl` |
| 7. Prop sizing | Every-start-day eval simulation at 1/2/3/5 micros vs coin flip, with your firm's rules | `config/prop_firms/*.yaml` |
| 8. Paper trade | Same strategy code + same fill rules, live data, Discord alerts | `run_paper.py` |
| 9. Live | **Not built yet, on purpose.** Broker-side bracket orders, daily loss stop, kill switch | phase 2 |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then fill in keys - never commit .env
pytest                                                # all tests should pass
```

## Try it on fake data first

These runs check that the tools behave. They say nothing about real markets.

```bash
python scripts/make_synthetic_data.py futures --years 8 --out data/synthetic_nq.parquet
python scripts/make_synthetic_data.py futures --years 8 --plant-edge --out data/synthetic_nq_edge.parquet
python run_futures_lab.py --data data/synthetic_nq.parquet      --is-end 2019-12-31   # pure noise -> DOES NOT TRADE
python run_futures_lab.py --data data/synthetic_nq_edge.parquet --is-end 2019-12-31   # hidden edge -> CANDIDATE
```

If the noise file ever comes out as a CANDIDATE, something is broken.

## Futures (NQ/MNQ)

```bash
python scripts/download_databento.py --start 2016-01-01 --end 2026-11-01   # shows the price, asks first
python run_futures_lab.py --data data/nq_1m.parquet --strategy orb \
    --params '{"rr": 2.0}' --hypothesis "early breakouts keep going; faders get squeezed" \
    --prop config/prop_firms/example_50k.yaml --sizes 1,2,3,5
```

Each run prints a side-by-side table, the prop sizing table, a PASS/FAIL
checklist and a verdict, and saves a report in `reports/`.

- **Before trusting any prop numbers,** copy `config/prop_firms/example_50k.yaml` and fill in your firm's exact rules: the drawdown type and whether it locks, the daily loss limit, any consistency rule, and the fee.
- **Set your real costs** in `config/instruments.yaml`.

### Paper trade ("the dealer")

```bash
python run_paper.py futures --replay data/nq_1m.parquet --from 2026-06-01   # test the plumbing
python run_paper.py futures --live --qty 2                                  # live Databento bars
```

Schedule the live command for 9:00 ET on weekdays (cron / Task Scheduler). It
stops itself at 16:05.

## Solana memecoins

```bash
python scripts/download_solana_history.py --pools 40        # free GeckoTerminal data (slow: rate-limited)
python run_solana_lab.py --data data/solana --strategy sol_momentum
python run_paper.py solana --config config/solana_paper.yaml
```

- **Safety screen.** Every coin must pass every check before the bot watches it:
  - mint authority revoked
  - freeze authority revoked
  - no dangerous Token-2022 extensions (transfer taxes, hooks, permanent delegate)
  - minimum liquidity and pool age
  - holder concentration under the limit
  - a quoted buy-then-sell round trip that doesn't lose too much
- **Paper fills** use live Jupiter quotes, which include real price impact. Positions are valued at what you could actually sell for, not the chart price.

**Read this before trading memecoins:**

- **Most memecoins go to zero, and many are scams.** The safety screen removes obvious traps, not all of them.
- **Survivorship bias.** Historical data mostly contains coins that survived, so backtests look better than reality. Trust the paper trader's numbers over the backtest.
- **Real swaps add costs paper can't see:** MEV/sandwich attacks, failed transactions and latency. Expect live results to be worse than paper.
- **Keys:** never put a wallet private key in this repo or in a chat. Live Solana trading (phase 2) will use a separate hot wallet that only ever holds money you can afford to lose.

## Adding a strategy

1. Create `proplab/strategies/my_strategy.py` with a `DEFAULT_PARAMS` dict and a `generate_signals(df, params)` function.
   - It returns a DataFrame with the columns `idx, side, stop_dist, target_dist, exit_time`.
   - `idx` is the bar whose **close** triggered the signal.
2. Register it in `proplab/strategies/__init__.py`.
3. Run a lab. The lookahead check runs every time and stops the run if the strategy peeks at future bars.

## Environment variables (`.env`)

| Name | Used for |
|---|---|
| `DATABENTO_API_KEY` | futures data (historical + live) |
| `DISCORD_WEBHOOK_URL` | trade alerts |
| `SOLANA_RPC_URL` | token safety checks (use a private RPC; the public one rate-limits hard) |
| `JUPITER_QUOTE_URL`, `JUPITER_API_KEY` | optional: override the Jupiter quote endpoint |

## Status

| Part | Status |
|---|---|
| Backtest engine, coin flip, prop sim, research log, futures replay paper trading | tested (`pytest`) |
| GeckoTerminal / Jupiter / Solana RPC / Databento clients | written to the documented APIs; response parsing is unit-tested, but they have **not yet been run against the live services**. Run each once and check the output. |
| Live order placement | not built (phase 2) |

Not financial advice. Trading futures and crypto can lose more than you expect.
