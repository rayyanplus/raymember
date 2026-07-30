"""Main CLI command entry point using standard Python argparse."""

import argparse
import json
import os
import sys
from typing import List, Optional

from raymember.sdk import Raymember
from raymember.storage.export_import import ExportImportEngine


def cli(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="raymember", description="Raymember: Persistent World-Memory Layer for AI Agents")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize a new local Raymember memory database")
    init_parser.add_argument("--db", default="raymember.db", help="Path to SQLite database file")
    init_parser.add_argument("--namespace", default="default", help="Initial namespace")

    # dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Start local web dashboard")
    dash_parser.add_argument("--db", default="raymember.db", help="Path to SQLite database file")
    dash_parser.add_argument("--port", type=int, default=8000, help="Port to run Uvicorn web server")
    dash_parser.add_argument("--host", default="127.0.0.1", help="Host address to bind server")

    # demo
    demo_parser = subparsers.add_parser("demo", help="Run interactive offline product demonstration")

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect entities and current state in a database")
    inspect_parser.add_argument("db_path", nargs="?", default="raymember.db", help="Path to SQLite database file")
    inspect_parser.add_argument("--namespace", default="default", help="Namespace to inspect")

    # query
    query_parser = subparsers.add_parser("query", help="Ask a natural language query against world memory")
    query_parser.add_argument("query_text", help="Question to ask (e.g. 'Where is the backpack?')")
    query_parser.add_argument("--db", default="raymember.db", help="Path to SQLite database file")
    query_parser.add_argument("--namespace", default="default", help="Namespace to query")

    # export
    export_parser = subparsers.add_parser("export", help="Export world memory state to JSON file")
    export_parser.add_argument("--db", default="raymember.db", help="Path to SQLite database file")
    export_parser.add_argument("--format", choices=["json", "text"], default="json", help="Export format")
    export_parser.add_argument("--output", help="Output file path (e.g. memory_export.json)")
    export_parser.add_argument("--namespace", help="Optional namespace filter")

    # import
    import_parser = subparsers.add_parser("import", help="Import world memory state from JSON file")
    import_parser.add_argument("--db", default="raymember.db", help="Target SQLite database file")
    import_parser.add_argument("--input", required=True, help="Input JSON file path (e.g. memory_export.json)")

    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return

    if parsed_args.command == "init":
        mem = Raymember(database_path=parsed_args.db, namespace=parsed_args.namespace)
        mem.close()
        print(f"Successfully initialized Raymember database at '{parsed_args.db}' (namespace: '{parsed_args.namespace}').")

    elif parsed_args.command == "dashboard":
        os.environ["RAYMEMBER_DB_PATH"] = parsed_args.db
        try:
            import uvicorn
            from raymember.dashboard.app import app
            print(f"Starting Raymember Local Web Dashboard on http://{parsed_args.host}:{parsed_args.port}")
            print(f"Connecting to SQLite Database: {parsed_args.db}")
            uvicorn.run(app, host=parsed_args.host, port=parsed_args.port, log_level="info")
        except ImportError:
            print("Error: FastAPI and Uvicorn are required for the dashboard. Install via:\n  pip install \"raymember[dashboard]\"")

    elif parsed_args.command == "demo":
        from examples.live_dashboard_demo import main as run_demo
        run_demo()

    elif parsed_args.command == "inspect":
        db_file = parsed_args.db_path
        if not os.path.exists(db_file):
            print(f"Error: Database file '{db_file}' does not exist.")
            return

        mem = Raymember(database_path=db_file, namespace=parsed_args.namespace)
        overview = mem.context("What is in memory?")
        print(f"=== Memory Database Inspection ({db_file} | Namespace: {parsed_args.namespace}) ===")
        print(overview)
        mem.close()

    elif parsed_args.command == "query":
        mem = Raymember(database_path=parsed_args.db, namespace=parsed_args.namespace)
        ans = mem.ask(parsed_args.query_text)
        print(f"Query: {parsed_args.query_text}")
        print(f"Answer: {ans.answer}")
        mem.close()

    elif parsed_args.command == "export":
        data = ExportImportEngine.export_to_json(db_path=parsed_args.db, output_path=parsed_args.output, namespace=parsed_args.namespace)
        if parsed_args.output:
            print(f"Successfully exported memory state to '{parsed_args.output}'.")
        else:
            print(json.dumps(data, indent=2))

    elif parsed_args.command == "import":
        count = ExportImportEngine.import_from_json(db_path=parsed_args.db, input_path=parsed_args.input)
        print(f"Successfully imported {count} entities into '{parsed_args.db}' from '{parsed_args.input}'.")


if __name__ == "__main__":
    cli()
