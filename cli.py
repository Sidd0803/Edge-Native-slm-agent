"""
Parts triage agent — interactive CLI entry point.

Natural language requests are routed through the LangGraph agent, which reasons
over the local ChromaDB catalog using two tools:
  - lookup_part: semantic search for parts described in plain language
  - check_stock: exact SKU lookup returning stock level, price, and supplier

The agent handles four cases:
  - Part in stock       → SKU, description, quantity, unit price
  - Part out of stock   → SKU, description, supplier, lead time in days
  - Part not found      → clear message, suggestion to search by category
  - Ambiguous request   → agent asks a clarifying question before calling a tool

Runtime catalog editing bypasses the agent entirely and writes directly to
ChromaDB via the memory layer:
  /add                               prompt for all fields, add a new part
  /update <sku> <field> <value>      update a single field on an existing part
  /remove <sku>                      delete a part from the catalog
  /list <category>                   list all parts in a category

Run with --verbose / -v to print tool calls and results as the agent reasons.
All inference and retrieval is local — no outbound network calls during operation.
"""

import sys

from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from agent.memory import PartsMemory

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv


def _final_response(messages: list) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "name", None):
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                return msg.content
    return "(no response)"


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _coerce(value: str):
    if value.lstrip("-").isdigit():
        return int(value)
    if _is_float(value):
        return float(value)
    return value


def handle_command(cmd: str, memory: PartsMemory):
    parts = cmd.split()
    command = parts[0]

    if command == "/add":
        print("Enter part details:")
        sku = input("  SKU: ").strip().upper()
        description = input("  Description: ").strip()
        category = input("  Category: ").strip().lower()
        stock_qty = int(input("  Stock qty: ").strip())
        unit_price = float(input("  Unit price: ").strip())
        supplier = input("  Supplier: ").strip()
        supplier_lead_days = int(input("  Lead days: ").strip())
        reorder_threshold = int(input("  Reorder threshold: ").strip())
        try:
            memory.add_part({
                "sku": sku,
                "description": description,
                "category": category,
                "stock_qty": stock_qty,
                "unit_price": unit_price,
                "supplier": supplier,
                "supplier_lead_days": supplier_lead_days,
                "reorder_threshold": reorder_threshold,
            })
            print(f"Added {sku}.\n")
        except ValueError as e:
            print(f"Error: {e}\n")

    elif command == "/update" and len(parts) >= 4:
        sku, field, value = parts[1].upper(), parts[2], parts[3]
        try:
            memory.update_part(sku, {field: _coerce(value)})
            print(f"Updated {sku}.{field} → {value}.\n")
        except (KeyError, ValueError) as e:
            print(f"Error: {e}\n")

    elif command == "/remove" and len(parts) >= 2:
        sku = parts[1].upper()
        try:
            memory.remove_part(sku)
            print(f"Removed {sku}.\n")
        except KeyError as e:
            print(f"Error: {e}\n")

    elif command == "/list" and len(parts) >= 2:
        category = " ".join(parts[1:]).lower()
        results = memory.list_by_category(category)
        if not results:
            print(f"No parts found in category '{category}'.\n")
        else:
            for p in results:
                stock_label = f"Qty: {p['stock_qty']}" if int(p['stock_qty']) > 0 else "OUT OF STOCK"
                print(f"  {p['sku']} — {p['description']} | {stock_label} | ${float(p['unit_price']):.2f}")
            print()

    else:
        print(f"Unknown command: {command}\n")


def run():
    print("Parts triage agent (local) — type a request or 'quit' to exit")
    print("Commands: /add  /update <sku> <field> <value>  /remove <sku>  /list <category>")
    if VERBOSE:
        print("Verbose mode ON — tool calls will be shown")
    print()

    graph = build_graph()
    memory = PartsMemory()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        if user_input.startswith("/"):
            handle_command(user_input, memory)
            continue

        result = graph.invoke({"messages": [HumanMessage(content=user_input)]})

        if VERBOSE:
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  [tool] {tc['name']}({tc['args']})")
                elif getattr(msg, "name", None):
                    print(f"  [result] {msg.content}")

        print(_final_response(result["messages"]))
        print()


if __name__ == "__main__":
    run()
