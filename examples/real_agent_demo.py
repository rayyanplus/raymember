"""
Raymember Phase 6 Example: Real Agent Integration Demo
Demonstrates supplying persistent, conflict-aware world-state context to an offline model.

No cloud dependencies or API keys required.
"""

import os
from typing import Callable, Dict, Any, List
from raymember import Raymember


class OfflineMockAgentModel:
    """
    Deterministic, offline mock LLM model interface matching model(prompt: str) -> str.
    Parses structured Raymember context or question to return consistent, ground-truth answers.
    """

    def __call__(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Check if prompt contains Raymember context with accepted ETA
        if "raymember world context" in prompt_lower or "accepted current state" in prompt_lower or "estimated_arrival" in prompt_lower:
            if "16:30" in prompt and ("17:15" in prompt or "conflicting" in prompt_lower):
                return (
                    "Based on Raymember's persistent world memory state for shipment_482, "
                    "the official estimated arrival time is 16:30 (confidence: 95%, source: tracking_api). "
                    "A lower-confidence conflicting report claiming 17:15 was rejected due to unverified provenance."
                )
            elif "16:30" in prompt:
                return "The estimated arrival time for shipment_482 is 16:30."

        # Default fallback for no-memory or uncontextualized questions
        if "when" in prompt_lower or "eta" in prompt_lower or "arrival" in prompt_lower:
            return "I do not have access to real-time memory or shipment tracking data for shipment_482."

        return "I am unable to answer without persistent world-state memory."


def main():
    db_path = "real_agent_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("=======================================================================")
    print("  Raymember Phase 6: Real Agent Integration & Behavioral Demo")
    print("=======================================================================")

    mem = Raymember(database_path=db_path)
    model = OfflineMockAgentModel()

    # 1. Store initial high-trust tracking observation from official carrier API
    print("\n[STEP 1] Ingesting High-Trust Tracking API Observation...")
    obs1 = mem.observe(
        entity="shipment_482",
        state={
            "status": "out_for_delivery",
            "driver": "driver_17",
            "destination": "Islamabad",
            "estimated_arrival": "16:30",
        },
        confidence=0.95,
        provenance="tracking_api",
    )
    print(f"  Observation ID: {obs1.observation_id}")
    print(f"  Raw State:      {obs1.state}")

    # 2. Ingest conflicting lower-trust observation
    print("\n[STEP 2] Ingesting Low-Trust Conflicting Observation...")
    obs2 = mem.observe(
        entity="shipment_482",
        state={
            "estimated_arrival": "17:15",
        },
        confidence=0.30,
        provenance="unreliable_sensor",
    )
    print(f"  Observation ID: {obs2.observation_id}")
    print(f"  Raw State:      {obs2.state}")

    # 3. Retrieve accepted current state & conflicting alternatives
    print("\n[STEP 3] Fetching Raymember World State...")
    state = mem.get("shipment_482")
    assert state is not None

    accepted_eta = state.current_attributes.get("estimated_arrival")
    print(f"\n--- Accepted Current State ---")
    print(f"  Entity:            {state.entity_label}")
    print(f"  Status:            {state.current_attributes.get('status')}")
    print(f"  Driver:            {state.current_attributes.get('driver')}")
    print(f"  Destination:       {state.current_attributes.get('destination')}")
    print(f"  Accepted ETA:      {accepted_eta} (Confidence: {state.confidence * 100:.0f}%, Provenance: {state.provenance})")

    eta_belief = state.attribute_beliefs.get("estimated_arrival", {})
    alternatives = eta_belief.get("alternative_values", [])

    print(f"\n--- Conflicting Alternatives ---")
    print(f"  Has Conflict:      {eta_belief.get('has_conflict')}")
    for alt in alternatives:
        print(f"  Alternative ETA:   {alt.get('value')} (Confidence: {alt.get('confidence') * 100:.0f}%, Provenance: {alt.get('provenance')})")

    assert accepted_eta == "16:30", f"Expected accepted ETA to remain '16:30', got '{accepted_eta}'"

    # 4. Generate Raymember context for LLM agent
    query_text = "When will shipment_482 arrive and what is its status?"
    print(f"\n[STEP 4] Exporting Raymember Context for Question: '{query_text}'...")
    context_str = mem.context(query_text)
    print("\n--- Generated Raymember Context ---")
    print(context_str)

    # 5. Execute Agent with Raymember Context vs No Memory Strategy
    print("\n[STEP 5] Comparing Agent Behavior...")

    # Strategy A: No Memory
    prompt_no_mem = f"Question: {query_text}\nAnswer:"
    ans_no_mem = model(prompt_no_mem)
    print(f"\n--- Strategy A (No Memory) ---")
    print(f"Agent Prompt: '{prompt_no_mem.strip()}'")
    print(f"Agent Answer: {ans_no_mem}")

    # Strategy C: Raymember Context
    prompt_with_mem = f"System Context:\n{context_str}\n\nQuestion: {query_text}\nAnswer:"
    ans_with_mem = model(prompt_with_mem)
    print(f"\n--- Strategy C (Raymember Context) ---")
    print(f"Agent Prompt: '{prompt_with_mem.strip()}'")
    print(f"Agent Answer: {ans_with_mem}")

    mem.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\n=======================================================================")
    print("  Real Agent Integration Demo Complete. All assertions verified!")
    print("=======================================================================")


if __name__ == "__main__":
    main()
