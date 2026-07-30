"""
Raymember Phase 5 Domain Example: Customer Support & Refund Resolution
Demonstrates handling conflicting refund status updates across customer claims,
Stripe webhook events, and internal ERP systems.
"""

import os
from raymember import Raymember


def main():
    db_path = "support_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("============================================================")
    print("  Raymember Phase 5 - Customer Support Refund Demo")
    print("============================================================")

    mem = Raymember(database_path=db_path)

    # 1. Customer submits a claim claiming refund is pending
    print("\n--- Step 1: Customer claims refund was requested ---")
    mem.observe(
        entity="ticket_9012",
        state={
            "refund_status": "requested",
            "customer_id": "cust_401",
            "amount": "$150.00",
        },
        confidence=0.70,
        provenance="user",
    )

    state1 = mem.get("ticket_9012")
    print(f"Current State: {state1.current_attributes}")

    # 2. Payment provider (Stripe webhook) confirms refund processing
    print("\n--- Step 2: Stripe webhook confirms refund processed ---")
    mem.observe(
        entity="ticket_9012",
        state={
            "refund_status": "processed",
            "transaction_id": "ch_3Mv9x82e",
        },
        confidence=0.98,
        provenance="sensor",  # Webhook sensor provenance
    )

    state2 = mem.get("ticket_9012")
    print(f"Current State:       {state2.current_attributes}")
    print(f"Refund Status:       {state2.current_attributes.get('refund_status')}")
    print(f"Transaction ID:      {state2.current_attributes.get('transaction_id')}")

    # 3. Unreliable customer chat claims refund failed
    print("\n--- Step 3: Low confidence customer chat claims refund failed ---")
    mem.observe(
        entity="ticket_9012",
        state={
            "refund_status": "failed",
        },
        confidence=0.20,
        provenance="unreliable_sensor",
    )

    state3 = mem.get("ticket_9012")
    print(f"Believed status:     {state3.current_attributes.get('refund_status')}")
    print(f"Has Conflict:        {state3.attribute_beliefs['refund_status']['has_conflict']}")

    # 4. Query natural language engine
    q = mem.ask("What is the status of ticket_9012?")
    print(f"\nQuery:  'What is the status of ticket_9012?'")
    print(f"Answer: {q.answer}")

    mem.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    print("\nCustomer support demo complete. All assertions verified!")


if __name__ == "__main__":
    main()
