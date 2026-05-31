from langchain_core.tools import tool

from agent.memory import PartsMemory

_memory: PartsMemory | None = None


def _get_memory() -> PartsMemory:
    global _memory
    if _memory is None:
        _memory = PartsMemory()
    return _memory


@tool
def lookup_part(query: str) -> str:
    """Search for parts by natural language description. Use when the user describes a part rather than giving a specific SKU."""
    results = _get_memory().search(query, n=3)
    if not results:
        return "No matching parts found."
    return "\n".join(
        f"SKU: {p['sku']} | {p['description']} | Category: {p['category']} | "
        f"Stock: {p['stock_qty']} | Price: ${float(p['unit_price']):.2f} | "
        f"Supplier: {p['supplier']} | Lead: {p['supplier_lead_days']}d"
        for p in results
    )


@tool
def check_stock(sku: str) -> str:
    """Look up a specific part by exact SKU code. Use when the user provides a SKU."""
    result = _get_memory().get_by_sku(sku.upper())
    if result is None:
        return f"SKU {sku} not found in catalog."
    stock_label = f"In stock: {result['stock_qty']} units" if int(result['stock_qty']) > 0 else "OUT OF STOCK"
    return (
        f"SKU: {result['sku']} | {result['description']} | {stock_label} | "
        f"Price: ${float(result['unit_price']):.2f} | "
        f"Supplier: {result['supplier']} | Lead time: {result['supplier_lead_days']} days"
    )
