"""FastAPI local web dashboard server with namespace selection, provenance, write panel, and context inspector."""

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from raymember.sdk import Raymember
from raymember.storage.export_import import ExportImportEngine

app = FastAPI(title="Raymember Local Dashboard", version="0.1.0")

DB_PATH = os.environ.get("RAYMEMBER_DB_PATH", "raymember.db")


class ObserveRequest(BaseModel):
    entity: str
    room: Optional[str] = "unknown"
    state: Optional[Dict[str, Any]] = None
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    z: Optional[float] = 0.0
    confidence: float = 0.90
    source: str = "user"
    provenance: str = "sensor"
    namespace: str = "default"
    attributes: Optional[Dict[str, Any]] = None


@app.get("/api/namespaces")
def get_namespaces():
    mem = Raymember(database_path=DB_PATH)
    try:
        ns_list = mem.list_namespaces()
        return {"namespaces": ns_list}
    finally:
        mem.close()


@app.get("/api/overview")
def get_overview(namespace: str = Query("default")):
    mem = Raymember(database_path=DB_PATH, namespace=namespace)
    try:
        with mem.db.session_scope() as session:
            from raymember.storage.models import CurrentStateModel, EntityModel, ObservationModel
            total_entities = session.query(EntityModel).filter_by(namespace=namespace).count()
            total_obs = session.query(ObservationModel).filter_by(namespace=namespace).count()
            current_states = session.query(CurrentStateModel).filter_by(namespace=namespace).all()

            uncertain_count = sum(1 for cs in current_states if cs.confidence < 0.6)
            avg_conf = float(sum(cs.confidence for cs in current_states) / len(current_states)) if current_states else 1.0

            return {
                "total_entities": total_entities,
                "active_entities": len(current_states),
                "total_observations": total_obs,
                "uncertain_entities": uncertain_count,
                "average_confidence": round(avg_conf, 2),
                "namespace": namespace,
                "database_path": DB_PATH,
            }
    finally:
        mem.close()


@app.get("/api/entities")
def get_entities(namespace: str = Query("default")):
    mem = Raymember(database_path=DB_PATH, namespace=namespace)
    try:
        with mem.db.session_scope() as session:
            from raymember.storage.models import CurrentStateModel, EntityModel
            entities = session.query(EntityModel).filter_by(namespace=namespace).all()
            results = []

            for ent in entities:
                state_res = mem.get(ent.canonical_name)
                if state_res:
                    results.append(state_res.to_dict())
            return {"entities": results}
    finally:
        mem.close()


@app.get("/api/timeline")
def get_timeline(namespace: str = Query("default"), limit: int = 20):
    mem = Raymember(database_path=DB_PATH, namespace=namespace)
    try:
        changes = mem.changes(limit=limit)
        return {"timeline": changes}
    finally:
        mem.close()


@app.get("/api/context-inspector")
def inspect_context(q: str = Query(...), namespace: str = Query("default")):
    mem = Raymember(database_path=DB_PATH, namespace=namespace)
    try:
        diag = mem.context_result(q)
        return {
            "query": diag.query,
            "formatted_context": diag.formatted_context,
            "selected_items": diag.selected_items,
            "relevance_scores": diag.relevance_scores,
            "truncated": diag.truncated,
            "total_relevance_score": diag.total_relevance_score,
        }
    finally:
        mem.close()


@app.get("/api/query")
def execute_query(q: str = Query(...), namespace: str = Query("default")):
    mem = Raymember(database_path=DB_PATH, namespace=namespace)
    try:
        ans = mem.ask(q)
        ctx = mem.context(q)
        return {
            "answer": ans.answer,
            "entity": ans.entity,
            "current_location": ans.current_location,
            "confidence": ans.belief_confidence,
            "state": ans.state,
            "context_summary": ctx,
        }
    finally:
        mem.close()


class AliasConfirmRequest(BaseModel):
    alias: str
    canonical: str
    namespace: str = "default"


class AliasRejectRequest(BaseModel):
    raw_location: str
    canonical: Optional[str] = None
    namespace: str = "default"


@app.get("/api/resolve-location")
def resolve_location_api(location: str = Query(...), namespace: str = Query("default")):
    mem = Raymember(database_path=DB_PATH, namespace=namespace)
    try:
        res = mem.resolve_location(location, namespace=namespace)
        return res.to_dict()
    finally:
        mem.close()


@app.post("/api/confirm-alias")
def confirm_alias_api(req: AliasConfirmRequest):
    mem = Raymember(database_path=DB_PATH, namespace=req.namespace)
    try:
        res = mem.confirm_location_alias(req.alias, req.canonical, namespace=req.namespace)
        return {"status": "success", "confirmed": res}
    finally:
        mem.close()


@app.post("/api/reject-alias")
def reject_alias_api(req: AliasRejectRequest):
    mem = Raymember(database_path=DB_PATH, namespace=req.namespace)
    try:
        res = mem.reject_location_resolution(req.raw_location, req.canonical, namespace=req.namespace)
        return {"status": "success", "rejected": res}
    finally:
        mem.close()


@app.post("/api/observe")
def post_observation(req: ObserveRequest):
    mem = Raymember(database_path=DB_PATH, namespace=req.namespace)
    try:
        record = mem.observe(
            entity=req.entity,
            location={"room": req.room, "x": req.x, "y": req.y, "z": req.z},
            attributes=req.attributes or {},
            confidence=req.confidence,
            source=req.source,
            provenance=req.provenance,
        )
        return {"status": "success", "observation": record.to_dict()}
    finally:
        mem.close()


@app.get("/api/export")
def export_json(namespace: Optional[str] = Query(None)):
    data = ExportImportEngine.export_to_json(DB_PATH, namespace=namespace)
    return JSONResponse(content=data)


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    static_html = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_html):
        with open(static_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Raymember Dashboard Loading...</h1>"
