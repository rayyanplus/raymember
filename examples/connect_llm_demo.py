"""
Raymember LLM Integration & Testing Demonstration.
Demonstrates connecting an LLM (Antigravity AI, Gemini, OpenAI, Ollama, Anthropic, or Mock)
to Raymember, querying persistent memory, and executing closed-loop state updates.
"""

import os
import sys
from raymember import Raymember, connect_llm


def main():
    db_path = "llm_connect_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("=======================================================================")
    print("  Raymember: Connect LLM & Interactive Memory Demonstration")
    print("=======================================================================")

    # 1. Initialize Raymember persistent memory database
    mem = Raymember(database_path=db_path)

    # 2. Connect LLM (using Antigravity AI assistant adapter)
    provider_choice = os.environ.get("RAYMEMBER_LLM_PROVIDER", "antigravity")
    print(f"\n[STEP 1] Connecting LLM using provider: '{provider_choice}'...")
    agent = connect_llm(memory=mem, provider=provider_choice)
    print(f"  Connected Agent: {agent.provider_name}")

    # 3. Ingest initial verified observations into Raymember
    print("\n[STEP 2] Ingesting Initial World State Observations...")
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
    print(f"  [Obs 1] shipment_482 estimated_arrival=16:30 (confidence: 95%, source: tracking_api)")

    obs2 = mem.observe(
        entity="toolkit",
        location={"room": "garage"},
        confidence=0.90,
        provenance="inventory_sensor",
    )
    print(f"  [Obs 2] toolkit location=garage (confidence: 90%, source: inventory_sensor)")

    # 4. Ingest conflicting low-trust report
    print("\n[STEP 3] Ingesting Conflicting Low-Trust Report...")
    obs3 = mem.observe(
        entity="shipment_482",
        state={"estimated_arrival": "17:15"},
        confidence=0.30,
        provenance="unreliable_sensor",
    )
    print(f"  [Obs 3] shipment_482 estimated_arrival=17:15 (confidence: 30%, source: unreliable_sensor)")

    # 5. Query LLM with Raymember Memory vs Baseline (No Memory)
    print("\n[STEP 4] Querying LLM with Persistent Memory Context...")
    query_text = "What is the status and estimated arrival time of shipment_482?"

    # Baseline: LLM without memory
    prompt_baseline = f"Question: {query_text}\nAnswer:"
    answer_baseline = agent.model(prompt_baseline)
    print("\n--- Strategy A: Baseline LLM (No Memory) ---")
    print(f"Response: {answer_baseline.strip()}")

    # Raymember Context: LLM with persistent memory
    answer_with_memory = agent.ask(query_text)
    print("\n--- Strategy C: Connected Raymember LLM Agent ---")
    print(f"Response:\n{answer_with_memory.strip()}")

    # 6. Closed-Loop Execution: User informs agent of movement, LLM updates Raymember
    print("\n[STEP 5] Executing Closed-Loop State Update...")
    user_statement = "The toolkit was moved from the garage to the workshop."
    print(f"  User Input: '{user_statement}'")

    answer_loop, new_observation = agent.run_and_remember(user_statement)
    print(f"  Agent Response: {answer_loop.strip()}")

    if new_observation:
        print(f"\n  [Raymember Auto-Update Recorded]")
        print(f"    Entity:      {new_observation['entity']}")
        print(f"    New Room:    {new_observation['room']}")
        print(f"    Confidence:  {new_observation['confidence'] * 100:.0f}%")
        print(f"    Provenance:  {new_observation['provenance']}")
        print(f"    Obs ID:      {new_observation['observation_id']}")

    # 7. Verify updated state in Raymember
    print("\n[STEP 6] Verifying Memory Belief State...")
    state_toolkit = mem.get("toolkit")
    assert state_toolkit is not None
    current_room = state_toolkit.current_location.get("room")
    print(f"  Toolkit Current Room in Raymember: '{current_room}'")
    assert current_room == "workshop", f"Expected 'workshop', got '{current_room}'"

    # Query LLM again to verify updated world state memory retrieval
    query_toolkit = "Where is the toolkit right now?"
    answer_updated = agent.ask(query_toolkit)
    print(f"\n--- Post-Update LLM Response ---")
    print(f"Query: '{query_toolkit}'")
    print(f"Response: {answer_updated.strip()}")

    # Clean up
    mem.close()
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\n=======================================================================")
    print("  LLM Connection & Testing Complete. All assertions verified!")
    print("=======================================================================")


if __name__ == "__main__":
    main()
