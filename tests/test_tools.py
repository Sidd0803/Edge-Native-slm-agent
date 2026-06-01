from unittest.mock import MagicMock, patch

from agent.tools import check_stock, lookup_part

IN_STOCK = {
    "sku": "HF-0001-A",
    "description": "Hydraulic fitting 3/8 inch NPT male",
    "category": "hydraulic fittings",
    "stock_qty": 50,
    "unit_price": 12.50,
    "supplier": "Parker Hannifin",
    "supplier_lead_days": 5,
    "reorder_threshold": 10,
}

OUT_OF_STOCK = {
    "sku": "PV-0002-B",
    "description": "Solenoid valve industrial grade",
    "category": "pneumatic valves",
    "stock_qty": 0,
    "unit_price": 149.99,
    "supplier": "SMC",
    "supplier_lead_days": 7,
    "reorder_threshold": 5,
}


# ---------------------------------------------------------------------------
# check_stock
# ---------------------------------------------------------------------------

def test_check_stock_in_stock():
    mock_mem = MagicMock()
    mock_mem.get_by_sku.return_value = IN_STOCK
    with patch("agent.tools._get_memory", return_value=mock_mem):
        result = check_stock.invoke({"sku": "HF-0001-A"})
    assert "In stock: 50 units" in result
    assert "HF-0001-A" in result
    assert "Parker Hannifin" in result
    assert "$12.50" in result


def test_check_stock_out_of_stock():
    mock_mem = MagicMock()
    mock_mem.get_by_sku.return_value = OUT_OF_STOCK
    with patch("agent.tools._get_memory", return_value=mock_mem):
        result = check_stock.invoke({"sku": "PV-0002-B"})
    assert "OUT OF STOCK" in result
    assert "SMC" in result
    assert "7 days" in result


def test_check_stock_not_found():
    mock_mem = MagicMock()
    mock_mem.get_by_sku.return_value = None
    with patch("agent.tools._get_memory", return_value=mock_mem):
        result = check_stock.invoke({"sku": "FAKE-0000-Z"})
    assert "not found" in result.lower()


def test_check_stock_uppercases_sku():
    mock_mem = MagicMock()
    mock_mem.get_by_sku.return_value = None
    with patch("agent.tools._get_memory", return_value=mock_mem):
        check_stock.invoke({"sku": "hf-0001-a"})
    mock_mem.get_by_sku.assert_called_once_with("HF-0001-A")


# ---------------------------------------------------------------------------
# lookup_part
# ---------------------------------------------------------------------------

def test_lookup_part_returns_multiple_results():
    mock_mem = MagicMock()
    mock_mem.search.return_value = [IN_STOCK, OUT_OF_STOCK]
    with patch("agent.tools._get_memory", return_value=mock_mem):
        result = lookup_part.invoke({"query": "hydraulic fitting"})
    assert "HF-0001-A" in result
    assert "PV-0002-B" in result


def test_lookup_part_no_results():
    mock_mem = MagicMock()
    mock_mem.search.return_value = []
    with patch("agent.tools._get_memory", return_value=mock_mem):
        result = lookup_part.invoke({"query": "conveyor belt"})
    assert "No matching parts found" in result


def test_lookup_part_out_of_stock_visible():
    mock_mem = MagicMock()
    mock_mem.search.return_value = [OUT_OF_STOCK]
    with patch("agent.tools._get_memory", return_value=mock_mem):
        result = lookup_part.invoke({"query": "valve"})
    assert "Stock: 0" in result


def test_lookup_part_searches_with_n3():
    mock_mem = MagicMock()
    mock_mem.search.return_value = [IN_STOCK]
    with patch("agent.tools._get_memory", return_value=mock_mem):
        lookup_part.invoke({"query": "fitting"})
    mock_mem.search.assert_called_once_with("fitting", n=3)
