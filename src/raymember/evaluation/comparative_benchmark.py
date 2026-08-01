"""
Comparative Memory Systems Benchmark Module for Raymember.

Primary Evaluation Objective: Factual Grounding & Hallucination Reduction
Evaluates four memory architectures:
  1. BASELINE_LLM (Zero memory, question only)
  2. FULL_CONTEXT_LLM (Unfiltered complete observation history)
  3. NAIVE_RETRIEVAL_LLM (Top-k raw observation retrieval via TF-IDF/lexical match)
  4. RAYMEMBER_LLM (Relevance-ranked, belief-resolved persistent world memory)

Measures Primary Grounding & Hallucination Metrics across 12 task categories:
  - Grounded answer accuracy
  - Supported claim rate
  - Unsupported claim rate
  - Contradiction rate
  - Hallucination rate
  - False-certainty rate
  - Correct abstention rate
  - Source attribution accuracy
  - Cross-turn consistency
"""

from dataclasses import asdict, dataclass, field
import json
import math
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from raymember.evaluation.harness import ModelHarness, DeterministicEvaluatorModel
from raymember.sdk import Raymember
from raymember.integrations.llm import connect_llm, RaymemberLLMAgent


@dataclass
class ComparativeScenario:
    """Structured benchmark scenario containing observations, question, and ground truth."""

    scenario_id: str
    category: str  # CURRENT_STATE, CONFLICT_RESOLUTION, TEMPORAL_REASONING, DISTRACTOR_RESISTANCE, CROSS_SESSION_PERSISTENCE, MISSING_INFO, CONTRADICTORY_EVIDENCE, UNCERTAIN_EVIDENCE, DISTRACTOR_HALLUCINATION, TEMPORAL_HALLUCINATION, ENTITY_CONFUSION, FALSE_PREMISE
    title: str
    observations: List[Dict[str, Any]]
    question: str
    expected_answer: str
    expected_confidence: Optional[float] = None
    expected_provenance: Optional[str] = None
    scale_size: int = 10
    is_abstention_expected: bool = False
    is_false_premise: bool = False


@dataclass
class TrialResult:
    """Outcome of evaluating a single scenario under one memory system."""

    scenario_id: str
    category: str
    system_name: str
    scale_size: int
    question: str
    expected_answer: str
    raw_model_response: str
    parsed_answer: str
    parsed_confidence: float
    parsed_reason: str
    grounded_answer_correct: bool
    supported_claim: bool
    unsupported_claim: bool
    is_contradiction: bool
    is_hallucination: bool
    false_certainty: bool
    correct_abstention: bool
    source_attribution_correct: bool
    context_char_count: int
    prompt_input_tokens: int
    prompt_output_tokens: int
    latency_ms: float
    deterministic_answer: bool = False
    llm_call_avoided: bool = False
    validation_failed: bool = False
    fallback_used: bool = False
    false_premise_corrected: bool = False
    entity_isolation_correct: bool = False
    temporal_gap_abstained: bool = False


class ComparativeScenarioGenerator:
    """Deterministic, seed-driven generator for synthetic grounding & hallucination benchmark scenarios."""

    ENTITIES = ["toolkit_A", "toolkit_B", "robot_alpha", "shipment_482", "laptop_m3", "container_77", "forklift_02", "drone_v9"]
    ROOMS = ["garage", "workshop", "attic", "lab_A", "warehouse_b", "loading_dock", "assembly_bay", "storage_room"]
    DISTRACTOR_ENTITIES = [f"sensor_node_{i:03d}" for i in range(1, 100)] + [f"pallet_{i:03d}" for i in range(100, 200)] + [f"crate_{i:03d}" for i in range(200, 300)]
    PROVENANCES = ["tracking_api", "camera_feed", "inventory_sensor", "manual_audit", "unreliable_sensor"]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_scenarios(
        self,
        num_scenarios: int = 120,
        scale_sizes: Optional[List[int]] = None,
    ) -> List[ComparativeScenario]:
        """Generates deterministic benchmark scenarios balanced across 12 grounding & memory categories."""
        self.rng = random.Random(self.seed)
        scales = scale_sizes or [50, 100, 250, 500, 1000]
        categories = [
            "CURRENT_STATE",
            "CONFLICT_RESOLUTION",
            "TEMPORAL_REASONING",
            "DISTRACTOR_RESISTANCE",
            "CROSS_SESSION_PERSISTENCE",
            "MISSING_INFO",
            "CONTRADICTORY_EVIDENCE",
            "UNCERTAIN_EVIDENCE",
            "DISTRACTOR_HALLUCINATION",
            "TEMPORAL_HALLUCINATION",
            "ENTITY_CONFUSION",
            "FALSE_PREMISE",
        ]

        scenarios: List[ComparativeScenario] = []
        base_time = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)

        for i in range(num_scenarios):
            cat = categories[i % len(categories)]
            scale = scales[i % len(scales)] if cat in ("DISTRACTOR_RESISTANCE", "DISTRACTOR_HALLUCINATION") else 10
            sc_id = f"sc_{self.seed}_{i+1:04d}_{cat.lower()}"

            if cat == "CURRENT_STATE":
                sc = self._generate_current_state(sc_id, base_time, i)
            elif cat == "CONFLICT_RESOLUTION":
                sc = self._generate_conflict_resolution(sc_id, base_time, i)
            elif cat == "TEMPORAL_REASONING":
                sc = self._generate_temporal_reasoning(sc_id, base_time, i)
            elif cat == "DISTRACTOR_RESISTANCE":
                sc = self._generate_distractor_resistance(sc_id, base_time, scale, i)
            elif cat == "CROSS_SESSION_PERSISTENCE":
                sc = self._generate_cross_session(sc_id, base_time, i)
            elif cat == "MISSING_INFO":
                sc = self._generate_missing_info(sc_id, base_time, i)
            elif cat == "CONTRADICTORY_EVIDENCE":
                sc = self._generate_contradictory_evidence(sc_id, base_time, i)
            elif cat == "UNCERTAIN_EVIDENCE":
                sc = self._generate_uncertain_evidence(sc_id, base_time, i)
            elif cat == "DISTRACTOR_HALLUCINATION":
                sc = self._generate_distractor_hallucination(sc_id, base_time, scale, i)
            elif cat == "TEMPORAL_HALLUCINATION":
                sc = self._generate_temporal_hallucination(sc_id, base_time, i)
            elif cat == "ENTITY_CONFUSION":
                sc = self._generate_entity_confusion(sc_id, base_time, i)
            else:  # FALSE_PREMISE
                sc = self._generate_false_premise(sc_id, base_time, i)

            scenarios.append(sc)

        return scenarios

    def _generate_current_state(self, sc_id: str, base_time: datetime, idx: int) -> ComparativeScenario:
        ent = self.ENTITIES[idx % len(self.ENTITIES)]
        r1, r2 = self.ROOMS[idx % len(self.ROOMS)], self.ROOMS[(idx + 3) % len(self.ROOMS)]
        t1, t2 = base_time.isoformat(), (base_time + timedelta(minutes=60)).isoformat()
        obs = [
            {"entity": ent, "location": {"room": r1}, "confidence": 0.90, "provenance": "inventory_sensor", "timestamp": t1},
            {"entity": ent, "location": {"room": r2}, "confidence": 0.95, "provenance": "camera_feed", "timestamp": t2},
        ]
        return ComparativeScenario(sc_id, "CURRENT_STATE", f"Location of {ent}", obs, f"Where is the {ent} located now?", r2, 0.95, "camera_feed", len(obs))

    def _generate_conflict_resolution(self, sc_id: str, base_time: datetime, idx: int) -> ComparativeScenario:
        ent = f"shipment_{idx+400}"
        eta1, eta2 = f"{14 + (idx % 4)}:30", f"{15 + (idx % 4)}:15"
        t1 = (base_time + timedelta(minutes=10)).isoformat()
        obs = [
            {"entity": ent, "state": {"estimated_arrival": eta1}, "confidence": 0.95, "provenance": "tracking_api", "timestamp": t1},
            {"entity": ent, "state": {"estimated_arrival": eta2}, "confidence": 0.30, "provenance": "unreliable_sensor", "timestamp": t1},
        ]
        return ComparativeScenario(sc_id, "CONFLICT_RESOLUTION", f"Conflicting ETA for {ent}", obs, f"What is the most reliable estimated arrival time for {ent}?", eta1, 0.95, "tracking_api", len(obs))

    def _generate_temporal_reasoning(self, sc_id: str, base_time: datetime, idx: int) -> ComparativeScenario:
        ent = self.ENTITIES[(idx + 1) % len(self.ENTITIES)]
        r1, r2, r3 = self.ROOMS[0], self.ROOMS[1], self.ROOMS[2]
        t1, t2, t3 = base_time.isoformat(), (base_time + timedelta(minutes=60)).isoformat(), (base_time + timedelta(minutes=120)).isoformat()
        obs = [
            {"entity": ent, "location": {"room": r1}, "confidence": 0.95, "provenance": "inventory_sensor", "timestamp": t1},
            {"entity": ent, "location": {"room": r2}, "confidence": 0.95, "provenance": "camera_feed", "timestamp": t2},
            {"entity": ent, "location": {"room": r3}, "confidence": 0.95, "provenance": "manual_audit", "timestamp": t3},
        ]
        q = f"Where was the {ent} before entering {r3}?" if idx % 2 == 0 else f"Where is the {ent} located now?"
        ans = r2 if idx % 2 == 0 else r3
        return ComparativeScenario(sc_id, "TEMPORAL_REASONING", f"Trajectory of {ent}", obs, q, ans, 0.95, None, len(obs))

    def _generate_distractor_resistance(self, sc_id: str, base_time: datetime, scale: int, idx: int) -> ComparativeScenario:
        ent, target_room = "target_item", "workshop"
        obs = [{"entity": ent, "location": {"room": "garage"}, "confidence": 0.90, "provenance": "inventory_sensor", "timestamp": base_time.isoformat()}]
        for d_i in range(scale - 2):
            d_ent = self.DISTRACTOR_ENTITIES[d_i % len(self.DISTRACTOR_ENTITIES)]
            d_room = self.ROOMS[d_i % len(self.ROOMS)]
            obs.append({"entity": d_ent, "location": {"room": d_room}, "confidence": 0.9, "provenance": "sensor", "timestamp": (base_time + timedelta(seconds=d_i+1)).isoformat()})
        obs.append({"entity": ent, "location": {"room": target_room}, "confidence": 0.95, "provenance": "camera_feed", "timestamp": (base_time + timedelta(minutes=500)).isoformat()})
        return ComparativeScenario(sc_id, "DISTRACTOR_RESISTANCE", f"Distractor Scale {scale}", obs, f"Where is the {ent} now?", target_room, 0.95, "camera_feed", scale)

    def _generate_cross_session(self, sc_id: str, base_time: datetime, idx: int) -> ComparativeScenario:
        ent, room = "laptop_m3", "lab_A"
        obs = [{"entity": ent, "location": {"room": room}, "confidence": 0.95, "provenance": "inventory_sensor", "timestamp": base_time.isoformat()}]
        return ComparativeScenario(sc_id, "CROSS_SESSION_PERSISTENCE", f"Session Persistence {ent}", obs, f"Which room is {ent} stored in according to memory?", room, 0.95, "inventory_sensor", len(obs))

    def _generate_missing_info(self, sc_id: str, base_time: datetime, idx: int) -> ComparativeScenario:
        ent = f"container_{idx+700}"
        obs = [{"entity": ent, "location": {"room": "garage"}, "confidence": 0.95, "provenance": "sensor", "timestamp": base_time.isoformat()}]
        return ComparativeScenario(sc_id, "MISSING_INFO", f"Missing Attribute for {ent}", obs, f"What is the serial number or weight of {ent}?", "unknown", 0.0, None, len(obs), is_abstention_expected=True)

    def _generate_contradictory_evidence(self, sc_id: str, base_time: datetime, idx: int) -> ComparativeScenario:
        ent = f"forklift_{idx+50}"
        obs = [
            {"entity": ent, "location": {"room": "loading_dock"}, "confidence": 0.95, "provenance": "tracking_api", "timestamp": base_time.isoformat()},
            {"entity": ent, "location": {"room": "attic"}, "confidence": 0.20, "provenance": "unreliable_sensor", "timestamp": base_time.isoformat()},
        ]
        return ComparativeScenario(sc_id, "CONTRADICTORY_EVIDENCE", f"Contradictory Location for {ent}", obs, f"Where is {ent} located according to verified high-trust evidence?", "loading_dock", 0.95, "tracking_api", len(obs))

    def _generate_uncertain_evidence(self, sc_id: str, base_time: datetime, idx: int) -> ComparativeScenario:
        ent = f"drone_{idx+10}"
        obs = [
            {"entity": ent, "location": {"room": "assembly_bay"}, "confidence": 0.51, "provenance": "sensor", "timestamp": base_time.isoformat()},
            {"entity": ent, "location": {"room": "storage_room"}, "confidence": 0.49, "provenance": "sensor", "timestamp": base_time.isoformat()},
        ]
        return ComparativeScenario(sc_id, "UNCERTAIN_EVIDENCE", f"Uncertain Location for {ent}", obs, f"Where is {ent} located?", "uncertain", 0.50, None, len(obs), is_abstention_expected=True)

    def _generate_distractor_hallucination(self, sc_id: str, base_time: datetime, scale: int, idx: int) -> ComparativeScenario:
        ent = "scanner_04"
        obs = [{"entity": ent, "location": {"room": "lab_A"}, "confidence": 0.95, "provenance": "sensor", "timestamp": base_time.isoformat()}]
        for d_i in range(scale - 1):
            obs.append({"entity": f"pallet_{d_i:03d}", "location": {"room": "warehouse_b"}, "state": {"contents": "chemical_vials"}, "confidence": 0.9, "provenance": "sensor", "timestamp": (base_time + timedelta(seconds=d_i+1)).isoformat()})
        return ComparativeScenario(sc_id, "DISTRACTOR_HALLUCINATION", f"Distractor Hallucination ({scale} obs)", obs, f"What contents or chemical vials are inside {ent}?", "unknown", 0.0, None, scale, is_abstention_expected=True)

    def _generate_temporal_hallucination(self, sc_id: str, base_time: datetime, idx: int) -> ComparativeScenario:
        ent = "robot_alpha"
        t_known = (base_time + timedelta(hours=5)).isoformat()
        t_query = (base_time - timedelta(hours=10)).strftime("%H:%M")
        obs = [{"entity": ent, "location": {"room": "workshop"}, "confidence": 0.95, "provenance": "camera_feed", "timestamp": t_known}]
        return ComparativeScenario(sc_id, "TEMPORAL_HALLUCINATION", f"Unobserved Timestamp for {ent}", obs, f"Where was {ent} located at {t_query} (before any observation)?", "unknown", 0.0, None, len(obs), is_abstention_expected=True)

    def _generate_entity_confusion(self, sc_id: str, base_time: datetime, idx: int) -> ComparativeScenario:
        obs = [
            {"entity": "toolkit_A", "location": {"room": "garage"}, "confidence": 0.95, "provenance": "camera_feed", "timestamp": base_time.isoformat()},
            {"entity": "toolkit_B", "location": {"room": "lab_A"}, "confidence": 0.95, "provenance": "inventory_sensor", "timestamp": base_time.isoformat()},
        ]
        return ComparativeScenario(sc_id, "ENTITY_CONFUSION", "Confusion between toolkit_A and toolkit_B", obs, "Where is toolkit_A located?", "garage", 0.95, "camera_feed", len(obs))

    def _generate_false_premise(self, sc_id: str, base_time: datetime, idx: int) -> ComparativeScenario:
        ent = "toolkit_A"
        obs = [
            {"entity": ent, "location": {"room": "garage"}, "confidence": 0.95, "provenance": "inventory_sensor", "timestamp": base_time.isoformat()},
            {"entity": ent, "location": {"room": "workshop"}, "confidence": 0.95, "provenance": "camera_feed", "timestamp": (base_time + timedelta(minutes=60)).isoformat()},
        ]
        return ComparativeScenario(sc_id, "FALSE_PREMISE", f"False Premise for {ent}", obs, f"Why was the {ent} moved from the kitchen?", "false_premise", 0.95, None, len(obs), is_false_premise=True)


class NaiveLexicalRetriever:
    """Top-k lexical TF-IDF / term-overlap retriever over raw observation records."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def retrieve(self, query: str, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not observations:
            return []
        q_terms = set(re.findall(r"\w+", query.lower()))
        scored_obs: List[Tuple[float, Dict[str, Any]]] = []
        for obs in observations:
            obs_str = json.dumps(obs).lower()
            obs_terms = set(re.findall(r"\w+", obs_str))
            score = len(q_terms.intersection(obs_terms)) / max(len(q_terms), 1)
            scored_obs.append((score, obs))
        scored_obs.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_obs[:self.top_k]]


class DeterministicScorer:
    """Deterministic ground-truth scoring for model JSON answers."""

    @staticmethod
    def parse_json_response(raw_text: str) -> Tuple[str, float, str]:
        raw_clean = raw_text.strip()
        if "{" in raw_clean and "}" in raw_clean:
            json_str = raw_clean[raw_clean.find("{"):raw_clean.rfind("}")+1]
            try:
                data = json.loads(json_str)
                ans = str(data.get("answer", "")).strip()
                conf = float(data.get("confidence", 0.0))
                reason = str(data.get("reason", "")).strip()
                return ans, conf, reason
            except Exception:
                pass
        return raw_clean.split("\n")[0].strip(), 0.0, ""

    @staticmethod
    def evaluate(
        scenario: ComparativeScenario,
        parsed_answer: str,
        parsed_confidence: float,
        parsed_reason: str,
        raw_response: str,
    ) -> Dict[str, bool]:
        exp_clean = scenario.expected_answer.strip().lower()
        ans_clean = parsed_answer.strip().lower()
        resp_lower = (parsed_answer + " " + parsed_reason + " " + raw_response).lower()

        is_abstention_resp = any(w in resp_lower for w in ["unknown", "unspecified", "no observation", "no record", "absent", "none", "uncertain", "not recorded"]) or len(ans_clean) == 0
        is_false_premise_resp = any(w in resp_lower for w in ["never", "false premise", "incorrect assumption", "not in kitchen", "did not move from kitchen", "invalid premise"])

        grounded_correct = False
        correct_abstention = False
        supported_claim = False
        unsupported_claim = False
        is_contradiction = False
        is_hallucination = False
        false_certainty = False

        if scenario.is_abstention_expected:
            if is_abstention_resp or exp_clean in ans_clean:
                grounded_correct = True
                correct_abstention = True
                supported_claim = True
            else:
                unsupported_claim = True
                is_hallucination = True
                if parsed_confidence >= 0.7:
                    false_certainty = True

        elif scenario.is_false_premise:
            if is_false_premise_resp or "false" in ans_clean:
                grounded_correct = True
                supported_claim = True
            else:
                unsupported_claim = True
                is_hallucination = True

        else:
            if (ans_clean == exp_clean) or (exp_clean in ans_clean) or (ans_clean in exp_clean and len(ans_clean) > 0):
                grounded_correct = True
                supported_claim = True
            else:
                unsupported_claim = True
                is_contradiction = True
                if len(ans_clean) > 0 and not is_abstention_resp:
                    is_hallucination = True

        if parsed_confidence >= 0.85 and not grounded_correct:
            false_certainty = True

        source_correct = True
        if scenario.expected_provenance:
            source_correct = scenario.expected_provenance.lower() in resp_lower

        return {
            "grounded_answer_correct": grounded_correct,
            "supported_claim": supported_claim,
            "unsupported_claim": unsupported_claim,
            "is_contradiction": is_contradiction,
            "is_hallucination": is_hallucination,
            "false_certainty": false_certainty,
            "correct_abstention": correct_abstention,
            "source_attribution_correct": source_correct,
        }


SYSTEM_PROMPT = (
    "You are a physical world state reasoning system. "
    "Use the provided context to answer the user question accurately. "
    "Respond STRICTLY in JSON format with keys:\n"
    '{\n  "answer": "<exact value or unknown/false_premise>",\n  "confidence": <0.0 to 1.0>,\n  "reason": "<short explanation>"\n}'
)


class MemoryBenchmarkRunner:
    """Executes comparative evaluation focused on Factual Grounding & Hallucination Reduction."""

    def __init__(
        self,
        provider: str = "mock",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        output_dir: str = "results",
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.output_dir = output_dir

        self.model_fn, self.provider_label, self.is_real = ModelHarness.get_model(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )

        self.retriever = NaiveLexicalRetriever(top_k=5)

    def run_benchmark(
        self,
        scenarios: List[ComparativeScenario],
        systems: Optional[List[str]] = None,
        num_runs: int = 1,
    ) -> Dict[str, Any]:
        target_systems = systems or ["baseline", "full_context", "naive_retrieval", "raymember"]

        os.makedirs(self.output_dir, exist_ok=True)
        db_dir = os.path.join(self.output_dir, "db_scratch")
        os.makedirs(db_dir, exist_ok=True)

        all_trials: List[TrialResult] = []

        for run_idx in range(num_runs):
            for sc in scenarios:
                for sys_name in target_systems:
                    trial = self._execute_trial(sc, sys_name, db_dir, run_idx)
                    all_trials.append(trial)

        aggregate_summary = self._aggregate_metrics(all_trials, scenarios, target_systems, num_runs)
        self._export_reports(all_trials, aggregate_summary)
        return aggregate_summary

    def _execute_trial(
        self,
        sc: ComparativeScenario,
        sys_name: str,
        db_dir: str,
        run_idx: int,
    ) -> TrialResult:
        db_path = os.path.join(db_dir, f"bench_{sc.scenario_id}_{sys_name}_{run_idx}.db")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass

        start_time = time.perf_counter()
        context_str = ""
        
        is_deterministic = False
        llm_avoided = False
        validation_failed = False
        fallback_used = False
        false_premise_corrected = False
        entity_isolation_correct = False
        temporal_gap_abstained = False

        if sys_name == "baseline":
            context_str = "No memory or prior observation context available."
            prompt = f"{SYSTEM_PROMPT}\n\n=== USER QUESTION ===\n{sc.question}\n"

        elif sys_name == "full_context":
            obs_lines = [json.dumps(o) for o in sc.observations]
            context_str = "\n".join(obs_lines)
            prompt = f"{SYSTEM_PROMPT}\n\n=== FULL OBSERVATION HISTORY ===\n{context_str}\n\n=== USER QUESTION ===\n{sc.question}\n"

        elif sys_name == "naive_retrieval":
            retrieved_obs = self.retriever.retrieve(sc.question, sc.observations)
            obs_lines = [json.dumps(o) for o in retrieved_obs]
            context_str = "\n".join(obs_lines)
            prompt = f"{SYSTEM_PROMPT}\n\n=== NAIVE RETRIEVED OBSERVATIONS ===\n{context_str}\n\n=== USER QUESTION ===\n{sc.question}\n"

        elif sys_name == "raymember":
            mem = Raymember(database_path=db_path)
            for obs in sc.observations:
                ent = obs.get("entity", "unknown")
                loc = obs.get("location")
                st = obs.get("state")
                conf = obs.get("confidence", 1.0)
                prov = obs.get("provenance", "sensor")
                ts = obs.get("timestamp")
                mem.observe(entity=ent, location=loc, state=st, confidence=conf, provenance=prov, timestamp=ts)

            if sc.category == "CROSS_SESSION_PERSISTENCE":
                mem.close()
                mem = Raymember(database_path=db_path)

            context_str = mem.context(sc.question)
            prompt = f"{SYSTEM_PROMPT}\n\n=== RAYMEMBER PERSISTENT MEMORY CONTEXT ===\n{context_str}\n\n=== USER QUESTION ===\n{sc.question}\n"
            mem.close()

        elif sys_name == "raymember_grounded":
            from raymember.grounding.config import GroundingConfig, GroundingMode
            from raymember.grounding.policy import GroundingPolicy
            
            mem = Raymember(database_path=db_path)
            for obs in sc.observations:
                ent = obs.get("entity", "unknown")
                loc = obs.get("location")
                st = obs.get("state")
                conf = obs.get("confidence", 1.0)
                prov = obs.get("provenance", "sensor")
                ts = obs.get("timestamp")
                mem.observe(entity=ent, location=loc, state=st, confidence=conf, provenance=prov, timestamp=ts)
            
            if sc.category == "CROSS_SESSION_PERSISTENCE":
                mem.close()
                mem = Raymember(database_path=db_path)
            
            config = GroundingConfig(mode=GroundingMode.STRICT)
            policy = GroundingPolicy(config=config)
            grounded_result = policy.evaluate_query(mem, sc.question)
            
            raw_resp = grounded_result.to_benchmark_json()
            context_str = grounded_result.answer
            prompt = f"Grounding query: {sc.question}"
            
            is_deterministic = grounded_result.deterministic
            llm_avoided = not grounded_result.llm_call_made
            validation_failed = (grounded_result.validation_status == "failed")
            fallback_used = grounded_result.fallback_used
            false_premise_corrected = (grounded_result.status.value == "contradicted_premise")
            temporal_gap_abstained = (grounded_result.status.value == "temporal_gap")
            
            mem.close()

        else:
            raise ValueError(f"Unknown system name '{sys_name}'")

        raw_resp = self.model_fn(prompt) if sys_name != "raymember_grounded" else raw_resp
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        parsed_ans, parsed_conf, parsed_reason = DeterministicScorer.parse_json_response(raw_resp)
        eval_scores = DeterministicScorer.evaluate(sc, parsed_ans, parsed_conf, parsed_reason, raw_resp)
        entity_isolation_correct = (sc.category == "ENTITY_CONFUSION" and eval_scores["grounded_answer_correct"])

        in_tokens = math.ceil(len(prompt) / 4.0)
        out_tokens = math.ceil(len(raw_resp) / 4.0)

        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass

        return TrialResult(
            scenario_id=sc.scenario_id,
            category=sc.category,
            system_name=sys_name,
            scale_size=sc.scale_size,
            question=sc.question,
            expected_answer=sc.expected_answer,
            raw_model_response=raw_resp,
            parsed_answer=parsed_ans,
            parsed_confidence=parsed_conf,
            parsed_reason=parsed_reason,
            grounded_answer_correct=eval_scores["grounded_answer_correct"],
            supported_claim=eval_scores["supported_claim"],
            unsupported_claim=eval_scores["unsupported_claim"],
            is_contradiction=eval_scores["is_contradiction"],
            is_hallucination=eval_scores["is_hallucination"],
            false_certainty=eval_scores["false_certainty"],
            correct_abstention=eval_scores["correct_abstention"],
            source_attribution_correct=eval_scores["source_attribution_correct"],
            context_char_count=len(context_str),
            prompt_input_tokens=in_tokens,
            prompt_output_tokens=out_tokens,
            latency_ms=latency_ms,
            deterministic_answer=is_deterministic,
            llm_call_avoided=llm_avoided,
            validation_failed=validation_failed,
            fallback_used=fallback_used,
            false_premise_corrected=false_premise_corrected,
            entity_isolation_correct=entity_isolation_correct,
            temporal_gap_abstained=temporal_gap_abstained,
        )

    def _aggregate_metrics(
        self,
        trials: List[TrialResult],
        scenarios: List[ComparativeScenario],
        systems: List[str],
        num_runs: int,
    ) -> Dict[str, Any]:

        summary: Dict[str, Any] = {
            "metadata": {
                "provider": self.provider_label,
                "is_real_model": self.is_real,
                "total_scenarios": len(scenarios),
                "num_runs": num_runs,
                "total_trials": len(trials),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "systems": {},
            "accuracy_by_category": {},
            "accuracy_by_scale": {},
        }

        for sys_name in systems:
            sys_trials = [t for t in trials if t.system_name == sys_name]
            n = float(len(sys_trials)) if sys_trials else 1.0

            summary["systems"][sys_name] = {
                "grounded_answer_accuracy": sum(1 for t in sys_trials if t.grounded_answer_correct) / n,
                "supported_claim_rate": sum(1 for t in sys_trials if t.supported_claim) / n,
                "unsupported_claim_rate": sum(1 for t in sys_trials if t.unsupported_claim) / n,
                "contradiction_rate": sum(1 for t in sys_trials if t.is_contradiction) / n,
                "hallucination_rate": sum(1 for t in sys_trials if t.is_hallucination) / n,
                "false_certainty_rate": sum(1 for t in sys_trials if t.false_certainty) / n,
                "correct_abstention_rate": sum(1 for t in sys_trials if t.correct_abstention) / n,
                "source_attribution_accuracy": sum(1 for t in sys_trials if t.source_attribution_correct) / n,
                "deterministic_answer_rate": sum(1 for t in sys_trials if t.deterministic_answer) / n,
                "llm_call_avoidance_rate": sum(1 for t in sys_trials if t.llm_call_avoided) / n,
                "validation_failure_rate": sum(1 for t in sys_trials if t.validation_failed) / n,
                "fallback_rate": sum(1 for t in sys_trials if t.fallback_used) / n,
                "false_premise_correction_rate": sum(1 for t in sys_trials if t.false_premise_corrected) / n,
                "entity_isolation_accuracy": sum(1 for t in sys_trials if t.entity_isolation_correct) / n,
                "temporal_gap_abstention_rate": sum(1 for t in sys_trials if t.temporal_gap_abstained) / n,
                "avg_latency_ms": sum(t.latency_ms for t in sys_trials) / n,
                "avg_input_tokens": sum(t.prompt_input_tokens for t in sys_trials) / n,
                "avg_context_chars": sum(t.context_char_count for t in sys_trials) / n,
            }

        # Category breakdown
        categories = sorted(list({t.category for t in trials}))
        for cat in categories:
            summary["accuracy_by_category"][cat] = {}
            for sys_name in systems:
                cat_sys_trials = [t for t in trials if t.category == cat and t.system_name == sys_name]
                if cat_sys_trials:
                    acc = sum(1 for t in cat_sys_trials if t.grounded_answer_correct) / float(len(cat_sys_trials))
                    summary["accuracy_by_category"][cat][sys_name] = acc

        # Scale breakdown (for DISTRACTOR_RESISTANCE and DISTRACTOR_HALLUCINATION)
        scales = sorted(list({t.scale_size for t in trials if t.category in ("DISTRACTOR_RESISTANCE", "DISTRACTOR_HALLUCINATION")}))
        for sz in scales:
            summary["accuracy_by_scale"][str(sz)] = {}
            for sys_name in systems:
                sz_trials = [t for t in trials if t.scale_size == sz and t.system_name == sys_name]
                if sz_trials:
                    acc = sum(1 for t in sz_trials if t.grounded_answer_correct) / float(len(sz_trials))
                    summary["accuracy_by_scale"][str(sz)][sys_name] = acc

        return summary

    def _export_reports(self, trials: List[TrialResult], summary: Dict[str, Any]) -> None:
        # CSV Export
        csv_path = os.path.join(self.output_dir, "benchmark_trials.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("scenario_id,category,system_name,scale_size,grounded_correct,supported,unsupported,contradiction,hallucination,false_certainty,correct_abstention,source_ok,latency_ms,input_tokens,deterministic,llm_avoided,validation_fail,fallback,false_premise,entity_isol,temporal_gap\n")
            for t in trials:
                f.write(f"{t.scenario_id},{t.category},{t.system_name},{t.scale_size},{t.grounded_answer_correct},{t.supported_claim},{t.unsupported_claim},{t.is_contradiction},{t.is_hallucination},{t.false_certainty},{t.correct_abstention},{t.source_attribution_correct},{t.latency_ms:.2f},{t.prompt_input_tokens},{t.deterministic_answer},{t.llm_call_avoided},{t.validation_failed},{t.fallback_used},{t.false_premise_corrected},{t.entity_isolation_correct},{t.temporal_gap_abstained}\n")

        # JSON Summary
        json_path = os.path.join(self.output_dir, "benchmark_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Markdown Report (Primary Grounding & Hallucination Focus)
        md_path = os.path.join(self.output_dir, "grounding_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Factual Grounding & Hallucination Reduction Report\n\n")
            f.write(f"- **Provider**: `{summary['metadata']['provider']}`\n")
            f.write(f"- **Scenarios**: {summary['metadata']['total_scenarios']}\n")
            f.write(f"- **Total Trials**: {summary['metadata']['total_trials']}\n\n")

            f.write("## Primary Grounding & Hallucination Metrics\n\n")
            f.write("| System | Grounded Acc | Supported Rate | Unsupported Rate | Contradiction Rate | Hallucination Rate | False Certainty | Abstention Acc |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")

            for sys_name, m in summary["systems"].items():
                f.write(
                    f"| `{sys_name}` | **{m['grounded_answer_accuracy']*100:.1f}%** | "
                    f"{m['supported_claim_rate']*100:.1f}% | {m['unsupported_claim_rate']*100:.1f}% | "
                    f"{m['contradiction_rate']*100:.1f}% | {m['hallucination_rate']*100:.1f}% | "
                    f"{m['false_certainty_rate']*100:.1f}% | {m['correct_abstention_rate']*100:.1f}% |\n"
                )

            f.write("\n## Secondary Engineering Metrics (Context & Latency)\n\n")
            f.write("| System | Avg Latency (ms) | Avg Input Tokens | Avg Context Chars |\n")
            f.write("| --- | --- | --- | --- |\n")
            for sys_name, m in summary["systems"].items():
                f.write(f"| `{sys_name}` | {m['avg_latency_ms']:.1f}ms | {m['avg_input_tokens']:.0f} | {m['avg_context_chars']:.0f} |\n")

            if "raymember_grounded" in summary["systems"]:
                f.write("\n## Grounding Guard Metrics\n\n")
                m = summary["systems"]["raymember_grounded"]
                f.write(f"- **Deterministic Answer Rate**: {m['deterministic_answer_rate']*100:.1f}%\n")
                f.write(f"- **LLM Call Avoidance Rate**: {m['llm_call_avoidance_rate']*100:.1f}%\n")
                f.write(f"- **Validation Failure Rate**: {m['validation_failure_rate']*100:.1f}%\n")
                f.write(f"- **Fallback Rate**: {m['fallback_rate']*100:.1f}%\n")
                f.write(f"- **False Premise Correction Rate**: {m['false_premise_correction_rate']*100:.1f}%\n")
                f.write(f"- **Entity Isolation Accuracy**: {m['entity_isolation_accuracy']*100:.1f}%\n")
                f.write(f"- **Temporal Gap Abstention Rate**: {m['temporal_gap_abstention_rate']*100:.1f}%\n")

            f.write("\n## Grounded Accuracy by Task Category\n\n")
            cats = list(summary["accuracy_by_category"].keys())
            sys_names = list(summary["systems"].keys())
            f.write("| Category | " + " | ".join(f"`{s}`" for s in sys_names) + " |\n")
            f.write("| --- | " + " | ".join("---" for _ in sys_names) + " |\n")
            for cat in cats:
                row = [cat]
                for s in sys_names:
                    val = summary["accuracy_by_category"].get(cat, {}).get(s, 0.0)
                    row.append(f"{val*100:.1f}%")
                f.write("| " + " | ".join(row) + " |\n")
