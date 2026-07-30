# Changelog

All notable changes to the Raymember project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-30

### Added
- **Phase 7 Real LLM Evaluation Harness (`src/raymember/evaluation/harness.py`)**: Provider-agnostic model harness supporting offline mock evaluation, OpenAI-compatible APIs, Ollama local models, Anthropic models, and custom callable functions (`model(prompt: str) -> str`).
- **32-Scenario Behavioral Benchmark (`src/raymember/evaluation/agent_comparison.py` & `examples/agent_comparison_benchmark.py`)**: Evaluates 3 context strategies across 12 requirement categories measuring accepted state accuracy, conflict interpretation accuracy, unsupported fact rate, contradiction rate, provenance citation accuracy, context token size, and latency.
- **Polished Developer Demo (`examples/developer_demo.py`)**: Interactive local terminal demo displaying real-time observation streaming, append-only evidence, accepted world state, attribute-level conflict badges, and side-by-side agent comparison.
- **Harness Integration Test Suite (`tests/test_real_model_harness.py` & `tests/test_agent_integration.py`)**: 11 new tests verifying adapter creation, offline mock execution, conflict formatting, and 32-scenario benchmark statistical calculations (100% offline).
- **Phase 5 Generalized World-State Engine**: Arbitrary structured state `state={...}`, per-attribute belief tracking, v4 SQLite schema migration, and 4 domain scripts (`logistics`, `customer_support`, `multiagent_task`, `robotics_spatial`).

## [0.2.0] - 2026-07-30

### Added
- **Semantic Location & Entity Resolution Engine**: Package `raymember.resolution` implementing deterministic normalization, configurable built-in & custom aliases (`washroom` -> `bathroom`, `livingroom` -> `living room`), and conservative offline fuzzy matching (`difflib`).
- **Confirmation & Rejection Persistence**: SDK methods `confirm_location_alias()` and `reject_location_resolution()` persisting user decisions in SQLite database across restarts.
- **Database Schema Migration v3**: Non-destructive column additions (`raw_location`, `normalized_location`, `canonical_location`, `resolution_method`, `resolution_confidence`, `resolution_confirmed`) and creation of `location_aliases` table.
- **Dashboard Resolution Integration**: Live location resolution preview in Write Panel with inline `[Confirm mapping]` and `[Keep as new location]` actions.
- **20-Scenario Resolution Test Suite**: `tests/test_resolution.py` covering exact, case, separator, alias, fuzzy, nonsense input, confirmation, rejection, persistence, and backward compatibility.
- **Interactive Resolution Demo**: `examples/location_resolution_demo.py` demonstrating all 9 resolution steps.

## [0.1.0] - 2026-07-29

### Added
- **Model-Agnostic Agent Integration Layer**: `MemoryAgent` and `OpenAICompatibleAgent` wrappers working with any callable model function without requiring cloud API keys.
- **Provenance & Write Safety**: Observation provenance tags (`user`, `sensor`, `tool`, `agent`, `inferred`, `imported`) with configurable trust maps protecting high-trust user observations.
- **Multi-Namespace Support**: Namespace scoping (`home`, `office`, `robot_1`) providing isolated memory partitions within a single database file.
- **JSON Import / Export Engine**: CLI `raymember export` and `raymember import` supporting full schema-validated state round-trips.
- **Database Schema Migration System**: Automatic SQLite backup creation and non-destructive column migrations.
- **Ranked Context Retrieval**: Relevance scoring based on entity match, room match, recency, confidence, and query terms, bounded by character limits with `context_result()` diagnostics.
- **30-Scenario Agent Integration Benchmark**: Deterministic evaluation framework comparing No Memory vs Naive Full Context vs Raymember Ranked Memory.
- **Local Web Dashboard v0.1.0**: FastAPI/Uvicorn single-page dashboard featuring Namespace dropdown, Provenance tags, Manual Write Panel, Context Inspector, and JSON Export.
- **Community Governance**: Added `SECURITY.md` and `CODE_OF_CONDUCT.md`.
