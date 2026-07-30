"""
Raymember Phase 5 Domain Example: Multi-Agent Task Coordination
Demonstrates resolving conflicting task ownership and completion states between agents.
"""

import os
from raymember import Raymember


def main():
    db_path = "multiagent_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("============================================================")
    print("  Raymember Phase 5 - Multi-Agent Task Coordination Demo")
    print("============================================================")

    mem = Raymember(database_path=db_path)

    # 1. Agent 1 assigns and takes ownership of task_17
    print("\n--- Step 1: Agent 1 claims task ownership ---")
    mem.observe(
        entity="task_17",
        state={
            "owner": "agent_alpha",
            "task_status": "in_progress",
            "assigned_priority": "high",
        },
        confidence=0.90,
        provenance="agent",
    )

    state1 = mem.get("task_17")
    print(f"Task Owner:  {state1.current_attributes.get('owner')}")
    print(f"Task Status: {state1.current_attributes.get('task_status')}")

    # 2. Agent 2 attempts to claim task_17 with lower confidence
    print("\n--- Step 2: Agent 2 attempts competing claim ---")
    mem.observe(
        entity="task_17",
        state={
            "owner": "agent_beta",
        },
        confidence=0.50,
        provenance="agent",
    )

    state2 = mem.get("task_17")
    print(f"Believed Owner: {state2.current_attributes.get('owner')}")
    print(f"Has Conflict:   {state2.attribute_beliefs['owner']['has_conflict']}")

    # 3. User manager explicitly confirms Agent Alpha ownership & completion
    print("\n--- Step 3: User manager confirms task completion ---")
    mem.observe(
        entity="task_17",
        state={
            "owner": "agent_alpha",
            "task_status": "completed",
            "result_artifact": "report_v1.pdf",
        },
        confidence=1.00,
        provenance="user",
    )

    state3 = mem.get("task_17")
    print(f"Final Owner:      {state3.current_attributes.get('owner')}")
    print(f"Final Status:     {state3.current_attributes.get('task_status')}")
    print(f"Artifact:         {state3.current_attributes.get('result_artifact')}")
    print(f"Transition Count: {len(state3.accepted_transitions)}")

    # 4. Natural language query
    q = mem.ask("Who currently owns task_17?")
    print(f"\nQuery:  'Who currently owns task_17?'")
    print(f"Answer: {q.answer}")

    mem.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    print("\nMulti-agent task coordination demo complete!")


if __name__ == "__main__":
    main()
