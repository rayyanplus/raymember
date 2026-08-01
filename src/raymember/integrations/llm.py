"""
Raymember LLM Connection Module.
Provides high-level helpers for connecting an LLM to Raymember and performing
closed-loop memory retrieval and state write-backs.
"""

import json
from typing import Any, Callable, Dict, Optional, Tuple, Union
from raymember.integrations.base import BaseAgentIntegration
from raymember.evaluation.harness import ModelHarness
from raymember.sdk import Raymember


class RaymemberLLMAgent(BaseAgentIntegration):
    """
    High-level agent integrating Raymember persistent world-memory with an LLM.
    Supports memory-augmented answering (`.ask`) and closed-loop memory updates (`.run_and_remember`).
    """

    def __init__(
        self,
        memory: Raymember,
        model_callable: Callable[[str], str],
        provider_name: str = "Antigravity AI",
        system_prompt: Optional[str] = None,
    ):
        super().__init__(memory)
        self.model = model_callable
        self.provider_name = provider_name
        self.system_prompt = system_prompt or (
            "You are an AI agent with access to persistent world memory managed by Raymember. "
            "Use the provided RAYMEMBER WORLD CONTEXT to answer the user accurately."
        )

    def run(
        self,
        user_input: str,
        max_context_items: int = 10,
        max_context_chars: int = 4000,
    ) -> str:
        """Retrieves memory context and generates an LLM response."""
        return self.ask(user_input, max_context_items=max_context_items, max_context_chars=max_context_chars)

    def ask(
        self,
        user_input: str,
        max_context_items: int = 10,
        max_context_chars: int = 4000,
    ) -> str:
        """Retrieves ranked context from Raymember memory and queries the LLM."""
        context_str = self.memory.context(
            query=user_input,
            max_items=max_context_items,
            max_characters=max_context_chars,
        )

        prompt = (
            f"{self.system_prompt}\n\n"
            f"=== RAYMEMBER WORLD CONTEXT ===\n"
            f"{context_str}\n\n"
            f"=== USER QUERY ===\n"
            f"{user_input}\n"
        )

        return self.model(prompt)

    def run_and_remember(
        self,
        user_input: str,
        confidence: float = 0.95,
        provenance: str = "user",
    ) -> Tuple[str, Optional[Dict[str, Any]]]:

        """
        Closed-loop memory execution:
        1. Queries LLM with Raymember memory context.
        2. Prompts LLM to extract any new entity observations or state updates from user statement.
        3. Persists extracted observations directly back into Raymember memory log.
        """
        # 1. Get standard LLM response
        answer = self.ask(user_input)

        # 2. Extract potential observation update
        extraction_prompt = (
            f"ACTION: EXTRACT_STATE_CHANGE\n"
            f"User input: {user_input}\n"
            f"Extract any physical state update or movement as JSON with keys 'entity' and 'room' if present, "
            f"or {{'status': 'no_change'}} if no state update occurred."
        )
        extraction_resp = self.model(extraction_prompt)

        recorded_obs = None
        try:
            if "entity" in extraction_resp and ("room" in extraction_resp or "location" in extraction_resp):
                # Clean JSON response
                clean_json = extraction_resp.strip()
                if "{" in clean_json and "}" in clean_json:
                    clean_json = clean_json[clean_json.find("{"):clean_json.rfind("}")+1]
                data = json.loads(clean_json)

                if "entity" in data and ("room" in data or "location" in data):
                    ent = data["entity"]
                    loc = data.get("room") or data.get("location")
                    loc_payload = {"room": str(loc)} if isinstance(loc, str) else loc
                    obs_record = self.memory.observe(
                        entity=ent,
                        location=loc_payload,
                        confidence=data.get("confidence", confidence),
                        provenance=provenance,
                    )
                    recorded_obs = {
                        "observation_id": obs_record.observation_id,
                        "entity": ent,
                        "room": obs_record.room,
                        "confidence": obs_record.confidence,
                        "provenance": obs_record.provenance if hasattr(obs_record, "provenance") else provenance,
                    }

        except Exception:
            pass  # Fallback if extraction fails gracefully



        return answer, recorded_obs


def connect_llm(
    memory: Raymember,
    provider: str = "antigravity",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    custom_model_fn: Optional[Callable[[str], str]] = None,
) -> RaymemberLLMAgent:
    """
    Connects an LLM to a Raymember persistent memory instance.

    Supported providers:
      - "antigravity" (default): Antigravity AI assistant with full memory reasoning capabilities
      - "gemini": Google Gemini API (gemini-2.5-flash)
      - "openai": OpenAI API or OpenAI-compatible server (vLLM, LM Studio)
      - "ollama": Local Ollama instance (http://localhost:11434)
      - "anthropic": Anthropic Claude API
      - "mock": Offline deterministic test model
    """
    model_fn, prov_label, _ = ModelHarness.get_model(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        custom_model_fn=custom_model_fn,
    )

    return RaymemberLLMAgent(
        memory=memory,
        model_callable=model_fn,
        provider_name=prov_label,
    )


class GroundedRaymemberAgent(RaymemberLLMAgent):
    """Grounded LLM agent with structured belief extraction, deterministic
    factual answers, deterministic abstention, uncertainty-aware responses,
    false-premise correction, and post-generation validation.

    Extends RaymemberLLMAgent with a grounding layer that intercepts queries,
    evaluates them against Raymember's structured belief state, and either:
    - Returns a deterministic answer (no LLM call) for direct factual queries
    - Validates LLM output against belief state for complex queries
    - Falls back to deterministic answer if validation fails

    Backward compatible: does not modify RaymemberLLMAgent behavior.
    """

    def __init__(
        self,
        memory: Raymember,
        model_callable: Callable[[str], str],
        provider_name: str = "Antigravity AI",
        system_prompt: Optional[str] = None,
        grounding_config: Optional[Any] = None,
    ):
        super().__init__(
            memory=memory,
            model_callable=model_callable,
            provider_name=provider_name,
            system_prompt=system_prompt,
        )
        from raymember.grounding.config import GroundingConfig
        from raymember.grounding.policy import GroundingPolicy
        from raymember.grounding.validator import GroundingValidator

        self.grounding_config = grounding_config or GroundingConfig()
        self.grounding_policy = GroundingPolicy(config=self.grounding_config)
        self.grounding_validator = GroundingValidator(config=self.grounding_config)

    def ask(
        self,
        user_input: str,
        max_context_items: int = 10,
        max_context_chars: int = 4000,
    ) -> str:
        """Grounded query answering.

        Flow:
        1. GroundingPolicy.evaluate_query() → GroundedResult
        2. If deterministic and STRICT mode → return deterministic answer
        3. Otherwise → call LLM, validate, potentially regenerate/fallback
        """
        result = self.ask_grounded(user_input, max_context_items, max_context_chars)
        return result.answer

    def ask_grounded(
        self,
        user_input: str,
        max_context_items: int = 10,
        max_context_chars: int = 4000,
    ):
        """Grounded query answering returning full GroundedResult.

        Returns:
            GroundedResult with grounding status, confidence, validation metadata.
        """
        from raymember.grounding.result import GroundedResult, GroundingStatus
        from raymember.grounding.config import GroundingMode

        # 1. Evaluate query against belief state
        grounded = self.grounding_policy.evaluate_query(self.memory, user_input)

        # 2. If deterministic and sufficient → return without LLM call
        if grounded.deterministic:
            if self.grounding_config.mode == GroundingMode.STRICT:
                return grounded
            elif grounded.status in (
                GroundingStatus.GROUNDED,
                GroundingStatus.INSUFFICIENT_EVIDENCE,
                GroundingStatus.CONTRADICTED_PREMISE,
                GroundingStatus.TEMPORAL_GAP,
            ):
                return grounded

        # 3. Need LLM reasoning — build prompt with belief context
        context_str = self.memory.context(
            query=user_input,
            max_items=max_context_items,
            max_characters=max_context_chars,
        )

        grounding_instruction = (
            "You are a physical world state reasoning system with access to "
            "Raymember persistent memory. Answer using ONLY the provided context. "
            "Do NOT invent facts, attributes, or temporal states not in the context. "
            "If information is missing, respond with 'unknown'. "
            "If a premise is false, correct it. "
            "Respond in JSON: {\"answer\": \"...\", \"confidence\": 0.0-1.0, \"reason\": \"...\"}"
        )

        prompt = (
            f"{grounding_instruction}\n\n"
            f"=== RAYMEMBER WORLD CONTEXT ===\n"
            f"{context_str}\n\n"
            f"=== USER QUERY ===\n"
            f"{user_input}\n"
        )

        # 4. Call LLM
        llm_response = self.model(prompt)
        grounded.llm_call_made = True

        # 5. Validate LLM output
        entity_state = None
        if grounded.entity:
            try:
                entity_state = self.memory.get(grounded.entity)
            except Exception:
                pass

        validation = self.grounding_validator.validate(
            llm_response=llm_response,
            entity_state=entity_state,
            grounded_result=grounded,
            entity_id=grounded.entity,
        )

        if validation.passed:
            grounded.validation_status = "passed"
            # Use LLM answer but preserve grounding metadata
            grounded.answer = llm_response
            return grounded

        # 6. Validation failed — attempt one regeneration
        grounded.validation_failures = validation.failures
        if self.grounding_config.max_regeneration_attempts > 0:
            stricter_prompt = (
                f"{grounding_instruction}\n\n"
                f"IMPORTANT: Your previous response failed validation. "
                f"Failures: {', '.join(validation.failures)}. "
                f"Answer ONLY from the provided context. Do NOT guess.\n\n"
                f"=== RAYMEMBER WORLD CONTEXT ===\n"
                f"{context_str}\n\n"
                f"=== USER QUERY ===\n"
                f"{user_input}\n"
            )
            retry_response = self.model(stricter_prompt)

            retry_validation = self.grounding_validator.validate(
                llm_response=retry_response,
                entity_state=entity_state,
                grounded_result=grounded,
                entity_id=grounded.entity,
            )

            if retry_validation.passed:
                grounded.validation_status = "passed"
                grounded.answer = retry_response
                return grounded

        # 7. Fallback to deterministic answer
        grounded.validation_status = "fallback"
        grounded.fallback_used = True
        # Keep the deterministic answer that was already set by the policy
        return grounded


def connect_llm_grounded(
    memory: Raymember,
    provider: str = "antigravity",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    custom_model_fn: Optional[Callable[[str], str]] = None,
    grounding_config: Optional[Any] = None,
) -> GroundedRaymemberAgent:
    """Connects a grounded LLM agent to a Raymember persistent memory instance.

    Creates a GroundedRaymemberAgent with structured belief extraction,
    deterministic factual answers, deterministic abstention, uncertainty-aware
    responses, false-premise correction, and post-generation validation.

    Supported providers: same as connect_llm().

    Args:
        memory: Raymember SDK instance.
        provider: Model provider name.
        model_name: Optional model name override.
        api_key: Optional API key.
        base_url: Optional base URL for compatible APIs.
        custom_model_fn: Optional custom model callable.
        grounding_config: Optional GroundingConfig instance.

    Returns:
        GroundedRaymemberAgent with grounding layer enabled.
    """
    model_fn, prov_label, _ = ModelHarness.get_model(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        custom_model_fn=custom_model_fn,
    )

    return GroundedRaymemberAgent(
        memory=memory,
        model_callable=model_fn,
        provider_name=prov_label,
        grounding_config=grounding_config,
    )
