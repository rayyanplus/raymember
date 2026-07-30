"""Relevance-ranked evidence retrieval and character-constrained context generation supporting compact, standard, and evidence modes."""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional
from raymember.validation.provenance import ProvenanceValidator


@dataclass
class RankedContextItem:
    entity_label: str
    location: Dict[str, Any]
    confidence: float
    source: str
    provenance: str
    timestamp: str
    relevance_score: float
    summary_text: str


@dataclass
class ContextResultDiagnostics:
    query: str
    formatted_context: str
    selected_items: List[Dict[str, Any]]
    relevance_scores: List[float]
    truncated: bool
    total_relevance_score: float
    mode: str = "standard"


class RankedContextRetriever:
    """Ranks evidence items by relevance score and formats character-bounded LLM prompt context."""

    @staticmethod
    def calculate_relevance(
        query: str,
        entity_label: str,
        room: str,
        confidence: float,
        provenance: str,
        timestamp: str,
        is_current_state: bool = False,
    ) -> float:
        q_lower = query.lower()
        ent_lower = entity_label.lower()
        room_lower = room.lower()

        score = 0.0

        # Entity match
        if ent_lower in q_lower or q_lower in ent_lower:
            score += 4.0

        # Room match
        if room_lower and room_lower in q_lower:
            score += 2.5

        # Current state preference
        if is_current_state:
            score += 2.0

        # Confidence & provenance trust multiplier
        prov_mult = ProvenanceValidator.get_trust_multiplier(provenance)
        score += float(confidence) * prov_mult * 1.5

        # Token overlap
        words = re.findall(r"\w+", q_lower)
        for w in words:
            if len(w) > 2 and (w in ent_lower or w in room_lower):
                score += 0.5

        return round(score, 4)

    @classmethod
    def generate_ranked_context(
        cls,
        query: str,
        candidate_items: List[Dict[str, Any]],
        max_items: int = 10,
        max_characters: int = 4000,
        mode: str = "standard",
    ) -> ContextResultDiagnostics:
        ranked_list: List[RankedContextItem] = []
        mode_clean = mode.lower() if mode else "standard"

        for item in candidate_items:
            ent = item.get("entity_label", "unknown")
            loc = item.get("location", {"room": item.get("room", "unknown")})
            room = loc.get("room", "unknown") if isinstance(loc, dict) else str(loc)
            conf = float(item.get("confidence", 1.0))
            src = item.get("source", "user")
            prov = item.get("provenance", "sensor")
            ts = item.get("timestamp", "")
            is_curr = item.get("is_current_state", False)

            score = cls.calculate_relevance(query, ent, room, conf, prov, ts, is_curr)

            attrs = item.get("attributes", {})
            attr_beliefs = item.get("attribute_beliefs", {})
            non_loc_attrs = {k: v for k, v in attrs.items() if k not in ("room", "location")}

            conflicts_summary = ""
            if attr_beliefs:
                conflict_parts = []
                for k, b in attr_beliefs.items():
                    if k in ("room", "location"):
                        continue
                    if b.get("has_conflict"):
                        alts = b.get("alternative_values", [])
                        alt_strs = [f"'{a['value']}' ({int(float(a.get('confidence', 0.5))*100)}% via {a.get('provenance', 'sensor')})" for a in alts]
                        conflict_parts.append(f"{k} had conflicting update(s): {', '.join(alt_strs)}")
                if conflict_parts:
                    conflicts_summary = " | CONFLICTS: " + "; ".join(conflict_parts)

            attr_str = f" | State: {non_loc_attrs}" if non_loc_attrs else ""
            loc_str = f" | Location: {room}" if room and room != "unknown" else ""

            if mode_clean in ("conflict_aware", "compact_conflict"):
                acc_lines = [f"ACCEPTED CURRENT STATE", f"{ent}:"]
                if room and room != "unknown":
                    acc_lines.append(f"- location: {room}")
                for k, v in non_loc_attrs.items():
                    acc_lines.append(f"- {k}: {v}")
                acc_lines.append(f"- provenance: {prov}")
                acc_lines.append(f"- confidence: {int(conf*100)}%")

                conf_lines = []
                if attr_beliefs:
                    for k, b in attr_beliefs.items():
                        if b.get("has_conflict"):
                            for a in b.get("alternative_values", []):
                                conf_lines.append(
                                    f"- {k}: {a.get('value')}\n  confidence: {int(float(a.get('confidence', 0.5))*100)}%\n  provenance: {a.get('provenance', 'sensor')}\n  status: rejected"
                                )
                if conf_lines:
                    summary = "\n".join(acc_lines) + "\n\nCONFLICTING ALTERNATIVES\n" + "\n".join(conf_lines)
                else:
                    summary = "\n".join(acc_lines)
            elif mode_clean == "compact":
                summary = f"- {ent}:{loc_str}{attr_str} (conf: {int(conf*100)}%, prov: {prov}){conflicts_summary}"
            elif mode_clean == "evidence":
                summary = f"- [OBSERVATION] Entity: {ent}{loc_str}{attr_str} | Conf: {int(conf*100)}% | Source: {src} | Prov: {prov} | Time: {ts}"
            else:  # standard
                is_curr_tag = "[CURRENT BELIEF]" if is_curr else "[OBSERVATION]"
                summary = f"- {is_curr_tag} Entity: {ent}{loc_str}{attr_str} | Conf: {int(conf*100)}% (prov={prov}) | Time: {ts}{conflicts_summary}"

            ranked_list.append(
                RankedContextItem(
                    entity_label=ent,
                    location=loc,
                    confidence=conf,
                    source=src,
                    provenance=prov,
                    timestamp=ts,
                    relevance_score=score,
                    summary_text=summary,
                )
            )

        # Sort by relevance score descending
        ranked_list.sort(key=lambda x: x.relevance_score, reverse=True)
        top_items = ranked_list[:max_items]

        lines = ["RAYMEMBER WORLD CONTEXT", f"Query: '{query}'", f"Mode: {mode_clean.upper()}", "Relevant World Beliefs & Evidence:"]
        selected_dicts = []
        scores = []
        truncated = False

        for item in top_items:
            candidate_line = item.summary_text
            current_length = sum(len(l) for l in lines) + len(candidate_line)
            if current_length > max_characters:
                truncated = True
                break

            lines.append(candidate_line)
            selected_dicts.append({
                "entity_label": item.entity_label,
                "location": item.location,
                "confidence": item.confidence,
                "provenance": item.provenance,
                "timestamp": item.timestamp,
                "relevance_score": item.relevance_score,
            })
            scores.append(item.relevance_score)

        formatted_text = "\n".join(lines)
        return ContextResultDiagnostics(
            query=query,
            formatted_context=formatted_text,
            selected_items=selected_dicts,
            relevance_scores=scores,
            truncated=truncated or len(ranked_list) > len(selected_dicts),
            total_relevance_score=sum(scores),
            mode=mode_clean,
        )
