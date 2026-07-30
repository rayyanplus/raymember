"""
Raymember Phase 5 Domain Example: Logistics & Package Tracking
Demonstrates handling conflicting tracking updates, per-attribute confidence,
and provenance trust (carrier tracking API vs driver observation).
"""

import os
from raymember import Raymember


def main():
    db_path = "logistics_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("============================================================")
    print("  Raymember Phase 5 - Logistics Domain Demo")
    print("============================================================")

    mem = Raymember(database_path=db_path)

    # 1. Initial tracking observation from official carrier API
    print("\n--- Step 1: Carrier API reports package out for delivery ---")
    obs1 = mem.observe(
        entity="delivery_4821",
        state={
            "status": "out_for_delivery",
            "driver": "driver_17",
            "destination": "Islamabad",
            "estimated_arrival": "2026-07-30T14:30:00",
        },
        confidence=0.95,
        provenance="tracking_api",
    )
    print(f"Observation ID:      {obs1.observation_id}")
    print(f"Recorded state:      {obs1.state}")

    state1 = mem.get("delivery_4821")
    print(f"Believed Attributes: {state1.current_attributes}")
    print(f"Believed status:     {state1.current_attributes.get('status')}")

    # 2. Conflicting estimated arrival from an unverified customer portal
    print("\n--- Step 2: Customer portal submits unverified delay claim ---")
    mem.observe(
        entity="delivery_4821",
        state={
            "estimated_arrival": "2026-07-30T18:00:00",
        },
        confidence=0.30,
        provenance="unreliable_sensor",
    )

    state2 = mem.get("delivery_4821")
    print(f"Believed Attributes: {state2.current_attributes}")
    print(f"Attribute Beliefs:   {state2.attribute_beliefs['estimated_arrival']}")
    print(f"Has Conflict:        {state2.attribute_beliefs['estimated_arrival']['has_conflict']}")

    # 3. Official delivery confirmation from driver sensor
    print("\n--- Step 3: Driver confirms successful delivery ---")
    mem.observe(
        entity="delivery_4821",
        state={
            "status": "delivered",
            "delivered_at": "2026-07-30T14:22:00",
        },
        confidence=0.99,
        provenance="sensor",
    )

    state3 = mem.get("delivery_4821")
    print(f"Final Status:        {state3.current_attributes.get('status')}")
    print(f"Driver:              {state3.current_attributes.get('driver')}")
    print(f"Transitions Log:     {len(state3.accepted_transitions)} accepted transition(s)")

    # 4. Natural language query
    print("\n--- Step 4: Natural Language Query ---")
    q1 = mem.ask("What is the status of delivery_4821?")
    print(f"Query:  'What is the status of delivery_4821?'")
    print(f"Answer: {q1.answer}")

    q2 = mem.ask("Why does Raymember believe delivery_4821 is delivered?")
    print(f"\nQuery:  'Why does Raymember believe delivery_4821 is delivered?'")
    print(f"Answer: {q2.answer}")

    mem.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    print("\nLogistics domain demo complete. All assertions verified!")


if __name__ == "__main__":
    main()
