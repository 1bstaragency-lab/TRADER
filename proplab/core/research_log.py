"""Append-only research log.

Every idea is written down BEFORE it is tested. Nothing is ever edited or
deleted: each line carries a hash of the line before it, so any edit breaks
the chain and `verify()` will say so. When something looks like a winner,
the lab tells you how many variations you tried to find it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class ResearchLog:
    def __init__(self, path: str = "research_log/log.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _records(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def _append(self, rec: dict) -> dict:
        recs = self._records()
        rec = {"seq": len(recs) + 1, "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
               "prev_hash": recs[-1]["hash"] if recs else "genesis", **rec}
        rec["hash"] = _h(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec

    def register(self, family: str, strategy: str, params: dict, hypothesis: str, data: str) -> dict:
        """Call BEFORE running a test."""
        vkey = _h({"family": family, "strategy": strategy, "params": params})[:12]
        return self._append({"type": "idea", "family": family, "strategy": strategy, "params": params,
                             "hypothesis": hypothesis, "data": data, "variation": vkey})

    def result(self, idea: dict, metrics: dict, verdict: str) -> dict:
        return self._append({"type": "result", "idea_seq": idea["seq"], "variation": idea["variation"],
                             "family": idea["family"], "metrics": metrics, "verdict": verdict})

    def holdout_unlock(self, idea: dict) -> dict:
        return self._append({"type": "holdout_unlock", "idea_seq": idea["seq"], "family": idea["family"],
                             "variation": idea["variation"]})

    def note(self, text: str) -> dict:
        return self._append({"type": "note", "text": text})

    def stats(self, family: str | None = None) -> dict:
        recs = self._records()
        ideas = [r for r in recs if r["type"] == "idea"]
        fam = [r for r in ideas if family is None or r["family"] == family]
        unlocks = [r for r in recs if r["type"] == "holdout_unlock" and (family is None or r["family"] == family)]
        return {"tests_run_total": len(ideas), "distinct_variations_total": len({r["variation"] for r in ideas}),
                "tests_in_family": len(fam), "distinct_in_family": len({r["variation"] for r in fam}),
                "holdout_unlocks_in_family": len(unlocks)}

    def verify(self) -> bool:
        prev = "genesis"
        for r in self._records():
            body = {k: v for k, v in r.items() if k != "hash"}
            if r["prev_hash"] != prev or _h(body) != r["hash"]:
                raise AssertionError(f"research log was edited or corrupted at entry #{r.get('seq')}")
            prev = r["hash"]
        return True
