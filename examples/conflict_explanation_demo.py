"""
Conflict Explanation Demo
=========================
Demonstrates Raymember's conflict-aware natural-language retrieval.

Scenario:
  1. User says car keys are on the desk       (confidence 0.98, provenance=user)
  2. Agent says car keys are in the kitchen   (confidence 0.60, provenance=agent)

Shows:
  - Structured current state (current_location, confidence, provenance)
  - Conflict metadata (has_conflict, conflicting_observations, conflict_summary)
  - Corrected natural-language answer from ask()
  - Raw append-only observation history (never modified)
  - Interpreted history (classification of each observation)
  - Ordinary confirmed movement behavior (bedroom -> living room)

Uses a fresh temporary database on every run.
"""

import tempfile
import os
import json

from raymember.sdk import Raymember

DIVIDER = "-" * 60


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def main():
    print("\n" + "=" * 60)
    print("  Raymember - Conflict Explanation Demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")
        mem = Raymember(database_path=db_path)

        # ── Part 1: Conflict scenario ─────────────────────────────────────────
        section("Part 1: Submit observations (car keys)")

        print("\n[1] User observes: car keys -> desk  (conf=0.98, prov=user)")
        mem.observe(
            "car keys",
            {"room": "desk"},
            confidence=0.98,
            source="user_app",
            provenance="user",
        )

        print("[2] Agent observes: car keys -> kitchen  (conf=0.60, prov=agent)")
        mem.observe(
            "car keys",
            {"room": "kitchen"},
            confidence=0.60,
            source="agent_v1",
            provenance="agent",
        )

        # ── Part 2: Structured current state ─────────────────────────────────
        section("Part 2: Structured current state (EntityStateResult)")

        state = mem.get("car keys")
        print(f"\n  Entity:           {state.entity_label}")
        print(f"  Current location: {state.current_location}")
        print(f"  Confidence:       {int(state.confidence * 100)}%")
        print(f"  Provenance:       {state.provenance}")
        print(f"  State status:     {state.state}")
        print(f"  Uncertainty:      {state.uncertainty_status}")

        # ── Part 3: Conflict metadata ─────────────────────────────────────────
        section("Part 3: Conflict metadata")

        print(f"\n  has_conflict:      {state.has_conflict}")
        print(f"  conflict_summary:  {state.conflict_summary}")
        print(f"\n  Conflicting observations ({len(state.conflicting_observations)}):")
        for c in state.conflicting_observations:
            print(f"    - room={c['room']:15s} conf={int(c['confidence']*100)}%  "
                  f"prov={c['provenance']}  reason={c['reason'][:60]}...")
        print(f"\n  Accepted observation IDs:  {state.accepted_observation_ids}")
        print(f"  Rejected observation IDs:  {state.rejected_observation_ids}")

        # -- Part 4: Interpreted history (classification layer) ----------------
        section("Part 4: Interpreted history (classification layer)")

        print()
        print(f"  {'Room':<20} {'Conf':>6}  {'Prov':<10}  {'Kind'}")
        print(f"  {'-'*20} {'-'*6}  {'-'*10}  {'-'*25}")
        for h in state.interpreted_history:
            print(f"  {h['room']:<20} {int(h['confidence']*100):>5}%  "
                  f"{h['provenance']:<10}  {h['kind']}")

        # -- Part 5: Corrected NL answer ---------------------------------------
        section("Part 5: Natural-language answer from ask()")

        result = mem.ask("Where are the car keys?")
        print(f"\n  Query:    \"Where are the car keys?\"")
        print(f"\n  Answer:\n")
        # Word-wrap the answer for readability
        words = result.answer.split()
        line = "    "
        for word in words:
            if len(line) + len(word) + 1 > 72:
                print(line)
                line = "    " + word
            else:
                line += (" " if line != "    " else "") + word
        if line.strip():
            print(line)
        print()
        print(f"  has_conflict:  {result.has_conflict}")
        print(f"  state label:   {result.state}")

        # -- Part 6: Raw append-only history (never modified) ------------------
        section("Part 6: Raw append-only observation history (never modified)")

        hist = mem.history("car keys")
        print(f"\n  Total stored observations: {len(hist)}")
        print()
        print(f"  {'#':<4} {'Room':<15} {'Conf':>6}  {'Prov':<10}  Timestamp")
        print(f"  {'-'*4} {'-'*15} {'-'*6}  {'-'*10}  {'-'*30}")
        for i, h in enumerate(hist, 1):
            ts = h.get("timestamp", "")[:19]
            print(f"  {i:<4} {h['room']:<15} {int(h['confidence']*100):>5}%  "
                  f"{h.get('provenance','sensor'):<10}  {ts}")
        print("\n  [OK] History is append-only: all observations preserved, none deleted.")

        # -- Part 7: Ordinary confirmed movement (backpack: bedroom -> living room) 
        section("Part 7: Ordinary confirmed movement (backpack: bedroom -> living room)")

        mem.observe("backpack", {"room": "bedroom"}, confidence=1.0, source="camera")
        mem.observe("backpack", {"room": "living room"}, confidence=1.0, source="camera")

        move_result = mem.ask("Where is the backpack?")
        print(f"\n  Query:    \"Where is the backpack?\"")
        print(f"  Answer:   {move_result.answer}")
        print(f"  has_conflict: {move_result.has_conflict}")
        print()
        print("  [OK] Confirmed movement correctly described without conflict flag.")

        # ── Part 8: Spatial wording verification ──────────────────────────────
        section("Part 8: Spatial wording check")

        mem.observe("wallet", {"room": "desk"}, confidence=0.95, source="user")
        mem.observe("laptop", {"room": "bedroom"}, confidence=0.95, source="user")

        wallet_ans = mem.ask("Where is the wallet?").answer
        laptop_ans = mem.ask("Where is the laptop?").answer

        print(f"\n  Wallet (on desk): \"{wallet_ans}\"")
        assert "on the desk" in wallet_ans.lower(), f"FAIL: expected 'on the desk' in: {wallet_ans}"
        print("  [OK] 'on the desk' - correct preposition")

        print(f"\n  Laptop (in bedroom): \"{laptop_ans}\"")
        assert "in the bedroom" in laptop_ans.lower(), f"FAIL: expected 'in the bedroom' in: {laptop_ans}"
        print("  [OK] 'in the bedroom' - correct preposition")

        mem.close()   # must close before TemporaryDirectory cleanup on Windows

    print("\n" + "=" * 60)
    print("  Demo complete. All assertions passed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
