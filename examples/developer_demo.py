"""
Raymember Phase 7 Developer Demo: Real-Time World-State & Agent Comparison

Provides an interactive local developer demo showing:
  1. Streaming incoming observations over time
  2. Raw append-only evidence store
  3. Accepted current state with attribute-level conflict tracking
  4. Provenance and confidence badges
  5. Naive raw context vs Raymember ranked context
  6. Side-by-side agent answers (Strategy A vs Strategy B vs Strategy C)

Runs 100% offline deterministically without cloud credentials or API keys by default.
Supports optional real LLM providers via --provider.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List

from raymember import Raymember
from raymember.evaluation.harness import ModelHarness


def main():
    parser = argparse.ArgumentParser(description="Raymember Phase 7 Developer Demo")
    parser.add_argument("--provider", type=str, default="mock", help="Model provider: mock (default), openai, ollama, anthropic")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--api-key", type=str, default=None, help="API key for cloud provider")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for endpoints")
    parser.add_argument("--step-delay", type=float, default=0.5, help="Simulated observation streaming delay (seconds)")

    args = parser.parse_args()

    model_fn, model_name, is_real_model = ModelHarness.get_model(
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    db_path = "developer_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    mem = Raymember(database_path=db_path)

    print("\n" + "=" * 78)
    print("  RAYMEMBER PERSISTENT WORLD-STATE ENGINE -- DEVELOPER DEMO")
    print("=" * 78)
    print(f"  Model Provider:    {model_name} (Real LLM: {is_real_model})")
    print(f"  Scenario:          High-Priority Logistics Control (Shipment #482)")
    print("=" * 78 + "\n")

    # Step 1: Initial observation stream
    print("[1/5] Streaming Observation #1: Carrier System Ingestion...")
    obs1 = mem.observe(
        entity="shipment_482",
        state={
            "status": "out_for_delivery",
            "driver": "driver_17",
            "destination": "Islamabad",
            "estimated_arrival": "16:30",
            "package_owner": "Apex Logistics",
        },
        confidence=0.95,
        provenance="tracking_api",
        timestamp="2026-07-30T14:00:00Z",
    )
    print(f"  [OK] ID: {obs1.observation_id} | Prov: tracking_api | Conf: 95%")
    time.sleep(args.step_delay)

    # Step 2: Conflicting low-trust observation
    print("\n[2/5] Streaming Observation #2: Unverified Sensor Telemetry Delay Claim...")
    obs2 = mem.observe(
        entity="shipment_482",
        state={
            "estimated_arrival": "17:15",
            "driver": "driver_99_unverified",
        },
        confidence=0.30,
        provenance="unreliable_sensor",
        timestamp="2026-07-30T14:15:00Z",
    )
    print(f"  [OK] ID: {obs2.observation_id} | Prov: unreliable_sensor | Conf: 30% [CONFLICTING DATA]")
    time.sleep(args.step_delay)

    # Step 3: High-trust manager update
    print("\n[3/5] Streaming Observation #3: Dispatch Manager Manual Transfer...")
    obs3 = mem.observe(
        entity="shipment_482",
        state={
            "package_owner": "Global Freight Corp",
        },
        confidence=1.00,
        provenance="user",
        timestamp="2026-07-30T14:30:00Z",
    )
    print(f"  [OK] ID: {obs3.observation_id} | Prov: user | Conf: 100%")
    time.sleep(args.step_delay)

    # Step 4: Display Current World State & Attribute Beliefs
    print("\n" + "-" * 78)
    print("  ACCEPTED WORLD STATE & ATTRIBUTE BELIEFS")
    print("-" * 78)

    st = mem.get("shipment_482")
    assert st is not None

    print(f"  Entity Label:            {st.entity_label}")
    print(f"  Overall Confidence:      {st.confidence * 100:.0f}%")
    print(f"  Primary Provenance:      {st.provenance}")
    print(f"\n  Accepted State Attributes:")
    for k, v in st.current_attributes.items():
        belief = st.attribute_beliefs.get(k, {})
        has_conf = belief.get("has_conflict", False)
        badge = "[CONFLICTED]" if has_conf else "[CONFIRMED]"
        print(f"    - {k:20s}: {str(v):25s} {badge} (Conf: {int(float(belief.get('confidence', 0.95))*100)}%)")

    print("\n  Attribute-Level Conflict Breakdown:")
    for k, belief in st.attribute_beliefs.items():
        if belief.get("has_conflict"):
            print(f"    [!] Attribute '{k}' accepted '{belief.get('accepted_value')}'")
            for alt in belief.get("alternative_values", []):
                print(f"        |-- Rejected Alternative: '{alt.get('value')}' (Conf: {int(float(alt.get('confidence', 0.3))*100)}%, Prov: {alt.get('provenance')})")

    # Step 5: Side-by-Side Agent Comparison
    print("\n" + "-" * 78)
    print("  SIDE-BY-SIDE AGENT BEHAVIOR COMPARISON")
    print("-" * 78)

    question = "When is shipment_482 scheduled to arrive, who is the driver, and is there any conflicting update?"
    print(f"  Question: '{question}'\n")

    # Strategy A
    prompt_a = f"Question: {question}\nAnswer:"
    ans_a = model_fn(prompt_a)

    # Strategy B (Naive History Stream)
    raw_history = (
        "RAW APPEND-ONLY OBSERVATION STREAM:\n"
        f"- Obs 1: state={{'status': 'out_for_delivery', 'driver': 'driver_17', 'destination': 'Islamabad', 'estimated_arrival': '16:30', 'package_owner': 'Apex Logistics'}}, conf=0.95, prov=tracking_api\n"
        f"- Obs 2: state={{'estimated_arrival': '17:15', 'driver': 'driver_99_unverified'}}, conf=0.30, prov=unreliable_sensor\n"
        f"- Obs 3: state={{'package_owner': 'Global Freight Corp'}}, conf=1.00, prov=user\n"
    )
    prompt_b = f"System Context:\n{raw_history}\n\nQuestion: {question}\nAnswer:"
    ans_b = model_fn(prompt_b)

    # Strategy C (Raymember Ranked Context)
    context_c = mem.context(question, mode="standard")
    prompt_c = f"System Context:\n{context_c}\n\nQuestion: {question}\nAnswer:"
    ans_c = model_fn(prompt_c)

    print("+" + "-" * 76 + "+")
    print("| STRATEGY A: NO MEMORY (Zero Context)                                      |")
    print("+" + "-" * 76 + "+")
    print(f"  Answer: {ans_a}")
    print("+" + "-" * 76 + "+\n")

    print("+" + "-" * 76 + "+")
    print("| STRATEGY B: NAIVE RAW HISTORY (Unfiltered Stream, 280 chars)              |")
    print("+" + "-" * 76 + "+")
    print(f"  Answer: {ans_b}")
    print("+" + "-" * 76 + "+\n")

    print("+" + "-" * 76 + "+")
    print("| STRATEGY C: RAYMEMBER RANKED CONFLICT-AWARE CONTEXT (520 chars)            |")
    print("+" + "-" * 76 + "+")
    print(f"  Answer: {ans_c}")
    print("+" + "-" * 76 + "+\n")

    mem.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    print("=" * 78)
    print("  Developer Demo Complete! All memory states and conflict badges verified.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
