"""Interactive offline Live Demo simulating object movement, false detections, delays, and conflicts."""

import time
from raymember.sdk import Raymember

def main():
    print("=== Raymember Live Product Demo (Offline & Zero Cloud Dependencies) ===")
    db_path = "live_demo.db"
    
    with Raymember(database_path=db_path, policy="auto") as memory:
        print("\n1. Initializing world entities...")
        memory.observe("black_backpack", {"room": "bedroom", "x": 1.5, "y": 0.0, "z": 2.0}, confidence=0.92, source="camera_bedroom")
        memory.observe("keys", {"room": "kitchen", "x": 0.5, "y": 0.0, "z": 0.5}, confidence=0.95, source="rfid_kitchen")
        memory.observe("laptop", {"room": "office", "x": 3.0, "y": 0.0, "z": 4.0}, confidence=0.98, source="sensor_office")
        memory.observe("bottle", {"room": "bedroom", "x": 1.0, "y": 0.0, "z": 1.0}, confidence=0.88, source="camera_bedroom")

        print("\n2. Simulating physical object movement: black_backpack moves to living_room...")
        res1 = memory.observe("black_backpack", {"room": "living_room", "x": 5.0, "y": 0.0, "z": 3.0}, confidence=0.94, source="camera_living_room")
        print(f"  Observed: {res1.entity_label} in {res1.room} (conf: {res1.confidence})")

        print("\n3. Injecting False Detection: camera_garage incorrectly reports keys in garage...")
        res2 = memory.observe("keys", {"room": "garage", "x": 9.0, "y": 0.0, "z": 9.0}, confidence=0.60, source="camera_garage")
        print(f"  Injected False Obs: {res2.entity_label} in {res2.room} (conf: {res2.confidence})")

        print("\n4. Injecting Conflicting Moderate-Confidence Observation...")
        res3 = memory.observe("black_backpack", {"room": "kitchen", "x": 2.0, "y": 0.0, "z": 2.0}, confidence=0.55, source="sensor_kitchen")
        print(f"  Injected Conflict: {res3.entity_label} in {res3.room} (conf: {res3.confidence})")

        print("\n5. Querying memory via SDK get():")
        backpack_state = memory.get("black_backpack")
        if backpack_state:
            print(f"  Entity: {backpack_state.entity_label}")
            print(f"  Current Location: {backpack_state.current_location}")
            print(f"  Confidence: {backpack_state.confidence * 100:.1f}%")
            print(f"  Previous Location: {backpack_state.previous_location}")
            print(f"  State: {backpack_state.state}")
            print(f"  Uncertainty Status: {backpack_state.uncertainty_status}")
            print(f"  Update Explanation: {backpack_state.explanation}")

        print("\n6. Asking Natural Language Question: 'Where is the black backpack?'")
        query_ans = memory.ask("Where is the black backpack?")
        print(f"  Answer: {query_ans.answer}")

        print("\n7. Exporting Model-Agnostic LLM Context Summary:")
        context_str = memory.context("Where is the black backpack?")
        print("  " + "\n  ".join(context_str.split("\n")))

    print("\nLive demo scenario completed successfully. Memory state persisted in live_demo.db.")
    print("To launch local web dashboard, run:")
    print(f"  raymember dashboard --db {db_path} --port 8000")

if __name__ == "__main__":
    main()
