"""
Location Resolution Demo
========================
Demonstrates Raymember's Phase 5 Semantic Entity & Location Resolution:
1. Store: android -> bathroom
2. Query: "What is in the washroom?"
3. Store: android -> livingroom
4. Resolves to: living room
5. Submit: shitlinger
6. Show that "shitlinger" is NOT silently mapped to "living room"
7. Demonstrate: confirm alias
8. Restart database connection
9. Show that confirmed aliases persist across process restarts

Uses a fresh temporary database on every run.
"""

import tempfile
import os
from raymember.sdk import Raymember

DIVIDER = "-" * 60


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def main():
    print("\n" + "=" * 60)
    print("  Raymember Phase 5 - Location Resolution Demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "resolution_demo.db")
        mem = Raymember(database_path=db_path)

        # ── Step 1: Store android -> bathroom ────────────────────────────────
        section("Step 1: Store android -> bathroom")
        rec1 = mem.observe("android", {"room": "bathroom"}, confidence=0.9, source="camera_1")
        print(f"  Stored entity:             {rec1.entity_label}")
        print(f"  Raw location:              {rec1.raw_location}")
        print(f"  Canonical location:        {rec1.canonical_location}")
        print(f"  Resolution method:         {rec1.resolution_method}")

        # ── Step 2: Query "What is in the washroom?" ───────────────────────────
        section("Step 2: Query 'What is in the washroom?'")
        res2 = mem.ask("What is in the washroom?")
        print(f"  Query:   \"What is in the washroom?\"")
        print(f"  Answer:  {res2.answer}")
        print(f"  Canonical location in state: {res2.current_location}")

        # ── Step 3 & 4: Store android -> livingroom -> resolves to "living room" ─
        section("Steps 3 & 4: Store android -> livingroom")
        rec3 = mem.observe("android", {"room": "livingroom"}, confidence=0.95, source="sensor_2")
        print(f"  Raw location:              {rec3.raw_location}")
        print(f"  Canonical location:        {rec3.canonical_location}")
        print(f"  Resolution method:         {rec3.resolution_method}")
        assert rec3.canonical_location == "living room", f"Expected 'living room', got '{rec3.canonical_location}'"
        print("  [OK] 'livingroom' correctly resolved to canonical 'living room'")

        # ── Step 5 & 6: Submit nonsense "shitlinger" ───────────────────────────
        section("Steps 5 & 6: Submit nonsense input 'shitlinger'")
        rec5 = mem.observe("android", {"room": "shitlinger"}, confidence=0.70, source="user")
        print(f"  Raw location:              {rec5.raw_location}")
        print(f"  Canonical location:        {rec5.canonical_location}")
        print(f"  Resolution method:         {rec5.resolution_method}")
        print(f"  Requires confirmation:     {not rec5.resolution_confirmed}")

        assert rec5.canonical_location != "living room", "FAIL: 'shitlinger' was incorrectly mapped to 'living room'!"
        assert rec5.canonical_location == "shitlinger", f"Expected 'shitlinger', got '{rec5.canonical_location}'"
        print("  [OK] 'shitlinger' preserved as NEW location without silent wrong merge.")

        # ── Step 7: Confirm alias "scullery" -> "kitchen" ─────────────────────
        section("Step 7: Demonstrate confirm alias ('scullery' -> 'kitchen')")
        confirm_res = mem.confirm_location_alias("scullery", "kitchen")
        print(f"  Confirmed alias mapping:   {confirm_res['alias']} -> {confirm_res['canonical']}")
        print(f"  Provenance:                {confirm_res['provenance']}")

        rec7 = mem.observe("toaster", {"room": "scullery"})
        print(f"  Observation raw room:      {rec7.raw_location}")
        print(f"  Resolved canonical room:   {rec7.canonical_location}")
        print(f"  Resolution method:         {rec7.resolution_method}")
        assert rec7.canonical_location == "kitchen"
        print("  [OK] User-confirmed alias 'scullery' resolved to 'kitchen'")

        mem.close()

        # ── Step 8 & 9: Restart DB connection and verify persistence ─────────
        section("Steps 8 & 9: Restart DB connection & verify alias persistence")
        mem_restarted = Raymember(database_path=db_path)

        res_check = mem_restarted.resolve_location("scullery")
        print(f"  Re-opened DB query for 'scullery':")
        print(f"    Canonical location:     {res_check.canonical_location}")
        print(f"    Resolution method:      {res_check.resolution_method}")
        assert res_check.canonical_location == "kitchen"
        assert res_check.resolution_method == "ALIAS"
        print("  [OK] Confirmed alias persisted across DB restarts!")

        mem_restarted.close()

    print("\n" + "=" * 60)
    print("  Demo complete. All Phase 5 assertions passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
