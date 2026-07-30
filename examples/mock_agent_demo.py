"""Mock Agent Demonstration working offline without external API keys or cloud LLMs."""

from raymember import Raymember
from raymember.integrations import MemoryAgent


def deterministic_mock_llm(prompt: str) -> str:
    """
    Deterministic mock language model function reading Raymember context
    and returning precise answers without requiring cloud API keys.
    """
    p_lower = prompt.lower()

    if "backpack" in p_lower:
        if "living" in p_lower or "bedroom" in p_lower:
            return "Based on Raymember context: The backpack is currently in the living room (previously in the bedroom)."
        return "The backpack is currently in the living room."

    if "kitchen" in p_lower:
        return "Based on Raymember context: The objects currently in the kitchen are keys and coffee_mug."

    if "changed" in p_lower or "recently" in p_lower:
        return "Based on Raymember context: Recent changes include the backpack moving from bedroom to living room."

    return "Based on Raymember context: Information retrieved successfully."


def main():
    print("=== Raymember Mock Agent Demonstration (Offline & Zero Cloud API Keys) ===\n")
    db_path = "mock_agent_demo.db"

    with Raymember(database_path=db_path, policy="auto") as memory:
        # Populate initial memory
        memory.observe("backpack", {"room": "bedroom"}, confidence=0.95, source="user", provenance="user")
        memory.observe("backpack", {"room": "living_room"}, confidence=0.92, source="camera_1", provenance="sensor")
        memory.observe("keys", {"room": "kitchen"}, confidence=0.90, source="rfid", provenance="sensor")
        memory.observe("coffee_mug", {"room": "kitchen"}, confidence=0.88, source="camera_kitchen", provenance="sensor")

        # Instantiate MemoryAgent
        agent = MemoryAgent(memory=memory, model=deterministic_mock_llm)

        queries = [
            "Where did I leave my backpack?",
            "Where was the backpack before?",
            "What objects are in the kitchen?",
            "What changed recently in world memory?",
        ]

        for q in queries:
            print(f"User Query: '{q}'")
            print("--- Retrieved Raymember Context ---")
            ctx = memory.context(q, max_items=5)
            print(ctx)
            print("--- Mock Model Response ---")
            resp = agent.run(q)
            print(f"Agent Response: {resp}\n" + "=" * 60 + "\n")

    print("Mock Agent demonstration completed successfully.")


if __name__ == "__main__":
    main()
