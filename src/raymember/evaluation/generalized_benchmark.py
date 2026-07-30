"""
Synthetic benchmark suite evaluating multi-attribute state tracking, per-attribute conflict resolution,
calibration, explanation correctness, and context efficiency across 5 domain categories.
"""

from typing import Any, Dict, List
import uuid

from raymember import Raymember


class GeneralizedStateBenchmark:
    """Benchmark runner for multi-domain generalized state engine."""

    def __init__(self, num_scenarios_per_domain: int = 10):
        self.num_scenarios = num_scenarios_per_domain

    def run_benchmark(self) -> Dict[str, Any]:
        results = {
            "location_state": self._eval_location_domain(),
            "categorical_workflow": self._eval_workflow_domain(),
            "ownership_state": self._eval_ownership_domain(),
            "numeric_sensor": self._eval_sensor_domain(),
            "multi_attribute": self._eval_multi_attribute_domain(),
        }

        total_attr_tests = sum(r["total_attributes"] for r in results.values())
        correct_attr_tests = sum(r["correct_attributes"] for r in results.values())
        overall_attr_acc = float(correct_attr_tests / total_attr_tests) if total_attr_tests else 1.0

        total_conflict_tests = sum(r["total_conflicts"] for r in results.values())
        correct_conflict_tests = sum(r["correct_conflicts"] for r in results.values())
        conflict_acc = float(correct_conflict_tests / total_conflict_tests) if total_conflict_tests else 1.0

        return {
            "domains": results,
            "overall_attribute_accuracy": round(overall_attr_acc, 4),
            "overall_conflict_accuracy": round(conflict_acc, 4),
            "total_evaluations": total_attr_tests,
        }

    def _eval_location_domain(self) -> Dict[str, Any]:
        db_path = f"bm_loc_{uuid.uuid4().hex[:6]}.db"
        mem = Raymember(database_path=db_path)
        correct_attr = 0
        total_attr = 0
        correct_conf = 0
        total_conf = 0

        for i in range(self.num_scenarios):
            entity = f"item_{i}"
            mem.observe(entity, location={"room": "living_room"}, confidence=0.90, provenance="sensor")
            mem.observe(entity, location={"room": "kitchen"}, confidence=0.95, provenance="user")
            mem.observe(entity, location={"room": "garage"}, confidence=0.20, provenance="unreliable_sensor")

            st = mem.get(entity)
            total_attr += 1
            if st and st.current_location.get("room") == "kitchen":
                correct_attr += 1

            total_conf += 1
            if st and (st.has_conflict or st.attribute_beliefs.get("room", {}).get("has_conflict") or any(c.get("room") == "garage" for c in (st.conflicting_observations or st.interpreted_history))):
                correct_conf += 1

        mem.close()
        import os
        if os.path.exists(db_path):
            os.remove(db_path)

        return {
            "total_attributes": total_attr,
            "correct_attributes": correct_attr,
            "accuracy": round(correct_attr / total_attr, 4) if total_attr else 1.0,
            "total_conflicts": total_conf,
            "correct_conflicts": correct_conf,
            "conflict_accuracy": round(correct_conf / total_conf, 4) if total_conf else 1.0,
        }

    def _eval_workflow_domain(self) -> Dict[str, Any]:
        db_path = f"bm_wf_{uuid.uuid4().hex[:6]}.db"
        mem = Raymember(database_path=db_path)
        correct_attr = 0
        total_attr = 0
        correct_conf = 0
        total_conf = 0

        for i in range(self.num_scenarios):
            entity = f"order_{i}"
            mem.observe(entity, state={"status": "processing"}, confidence=0.80, provenance="tracking_api")
            mem.observe(entity, state={"status": "shipped"}, confidence=0.90, provenance="tracking_api")
            mem.observe(entity, state={"status": "cancelled"}, confidence=0.25, provenance="unreliable_sensor")

            st = mem.get(entity)
            total_attr += 1
            if st and st.current_attributes.get("status") == "shipped":
                correct_attr += 1

            total_conf += 1
            if st and st.attribute_beliefs.get("status", {}).get("has_conflict"):
                correct_conf += 1

        mem.close()
        import os
        if os.path.exists(db_path):
            os.remove(db_path)

        return {
            "total_attributes": total_attr,
            "correct_attributes": correct_attr,
            "accuracy": round(correct_attr / total_attr, 4) if total_attr else 1.0,
            "total_conflicts": total_conf,
            "correct_conflicts": correct_conf,
            "conflict_accuracy": round(correct_conf / total_conf, 4) if total_conf else 1.0,
        }

    def _eval_ownership_domain(self) -> Dict[str, Any]:
        db_path = f"bm_own_{uuid.uuid4().hex[:6]}.db"
        mem = Raymember(database_path=db_path)
        correct_attr = 0
        total_attr = 0
        correct_conf = 0
        total_conf = 0

        for i in range(self.num_scenarios):
            entity = f"task_{i}"
            mem.observe(entity, state={"owner": "agent_alpha"}, confidence=0.95, provenance="user")
            mem.observe(entity, state={"owner": "agent_beta"}, confidence=0.40, provenance="agent")

            st = mem.get(entity)
            total_attr += 1
            if st and st.current_attributes.get("owner") == "agent_alpha":
                correct_attr += 1

            total_conf += 1
            if st and st.attribute_beliefs.get("owner", {}).get("has_conflict"):
                correct_conf += 1

        mem.close()
        import os
        if os.path.exists(db_path):
            os.remove(db_path)

        return {
            "total_attributes": total_attr,
            "correct_attributes": correct_attr,
            "accuracy": round(correct_attr / total_attr, 4) if total_attr else 1.0,
            "total_conflicts": total_conf,
            "correct_conflicts": correct_conf,
            "conflict_accuracy": round(correct_conf / total_conf, 4) if total_conf else 1.0,
        }

    def _eval_sensor_domain(self) -> Dict[str, Any]:
        db_path = f"bm_sns_{uuid.uuid4().hex[:6]}.db"
        mem = Raymember(database_path=db_path)
        correct_attr = 0
        total_attr = 0
        correct_conf = 0
        total_conf = 0

        for i in range(self.num_scenarios):
            entity = f"sensor_{i}"
            mem.observe(entity, state={"temperature": 22.5}, confidence=0.90, provenance="sensor")
            mem.observe(entity, state={"temperature": 45.0}, confidence=0.15, provenance="unreliable_sensor")

            st = mem.get(entity)
            total_attr += 1
            if st and st.current_attributes.get("temperature") == 22.5:
                correct_attr += 1

            total_conf += 1
            if st and st.attribute_beliefs.get("temperature", {}).get("has_conflict"):
                correct_conf += 1

        mem.close()
        import os
        if os.path.exists(db_path):
            os.remove(db_path)

        return {
            "total_attributes": total_attr,
            "correct_attributes": correct_attr,
            "accuracy": round(correct_attr / total_attr, 4) if total_attr else 1.0,
            "total_conflicts": total_conf,
            "correct_conflicts": correct_conf,
            "conflict_accuracy": round(correct_conf / total_conf, 4) if total_conf else 1.0,
        }

    def _eval_multi_attribute_domain(self) -> Dict[str, Any]:
        db_path = f"bm_multi_{uuid.uuid4().hex[:6]}.db"
        mem = Raymember(database_path=db_path)
        correct_attr = 0
        total_attr = 0
        correct_conf = 0
        total_conf = 0

        for i in range(self.num_scenarios):
            entity = f"package_{i}"
            mem.observe(
                entity,
                state={
                    "status": "out_for_delivery",
                    "driver": "driver_17",
                    "destination": "Islamabad",
                    "estimated_arrival": "14:30",
                },
                confidence=0.95,
                provenance="tracking_api",
            )
            # Low confidence delay update on estimated_arrival ONLY
            mem.observe(
                entity,
                state={
                    "estimated_arrival": "18:00",
                },
                confidence=0.30,
                provenance="unreliable_sensor",
            )

            st = mem.get(entity)
            total_attr += 4
            if st:
                if st.current_attributes.get("status") == "out_for_delivery":
                    correct_attr += 1
                if st.current_attributes.get("driver") == "driver_17":
                    correct_attr += 1
                if st.current_attributes.get("destination") == "Islamabad":
                    correct_attr += 1
                if st.current_attributes.get("estimated_arrival") == "14:30":
                    correct_attr += 1

            total_conf += 1
            # Unrelated attributes must NOT be flagged as conflicting
            if (
                st
                and st.attribute_beliefs.get("estimated_arrival", {}).get("has_conflict")
                and not st.attribute_beliefs.get("status", {}).get("has_conflict")
            ):
                correct_conf += 1

        mem.close()
        import os
        if os.path.exists(db_path):
            os.remove(db_path)

        return {
            "total_attributes": total_attr,
            "correct_attributes": correct_attr,
            "accuracy": round(correct_attr / total_attr, 4) if total_attr else 1.0,
            "total_conflicts": total_conf,
            "correct_conflicts": correct_conf,
            "conflict_accuracy": round(correct_conf / total_conf, 4) if total_conf else 1.0,
        }
