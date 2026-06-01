import pytest
from tests.conftest import SAMPLE_PARTS


# ---------------------------------------------------------------------------
# get_by_sku
# ---------------------------------------------------------------------------

def test_get_by_sku_found(temp_memory):
    result = temp_memory.get_by_sku("TEST-0001-A")
    assert result is not None
    assert result["sku"] == "TEST-0001-A"
    assert result["stock_qty"] == 50


def test_get_by_sku_not_found(temp_memory):
    assert temp_memory.get_by_sku("FAKE-0000-Z") is None


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_returns_list(temp_memory):
    results = temp_memory.search("hydraulic fitting", n=2)
    assert isinstance(results, list)
    assert len(results) > 0


def test_search_respects_n(temp_memory):
    results = temp_memory.search("valve", n=2)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# add_part
# ---------------------------------------------------------------------------

def test_add_part_and_retrieve(temp_memory):
    new_part = {
        "sku": "TEST-9999-X",
        "description": "New test manifold block",
        "category": "hydraulic fittings",
        "stock_qty": 10,
        "unit_price": 5.0,
        "supplier": "Test Supplier",
        "supplier_lead_days": 3,
        "reorder_threshold": 5,
    }
    temp_memory.add_part(new_part)
    result = temp_memory.get_by_sku("TEST-9999-X")
    assert result is not None
    assert result["description"] == "New test manifold block"


def test_add_duplicate_sku_raises(temp_memory):
    duplicate = dict(SAMPLE_PARTS[0])  # TEST-0001-A already exists
    with pytest.raises(ValueError, match="already exists"):
        temp_memory.add_part(duplicate)


# ---------------------------------------------------------------------------
# update_part
# ---------------------------------------------------------------------------

def test_update_existing_field(temp_memory):
    temp_memory.update_part("TEST-0001-A", {"stock_qty": 0})
    result = temp_memory.get_by_sku("TEST-0001-A")
    assert result["stock_qty"] == 0


def test_update_unknown_sku_raises(temp_memory):
    with pytest.raises(KeyError, match="not found"):
        temp_memory.update_part("FAKE-0000-Z", {"stock_qty": 5})


def test_update_unknown_field_raises(temp_memory):
    with pytest.raises(KeyError, match="Unknown field"):
        temp_memory.update_part("TEST-0001-A", {"qty": 5})  # correct name is stock_qty


def test_update_wrong_type_raises(temp_memory):
    # stock_qty is int; passing a non-numeric string should raise ValueError
    with pytest.raises(ValueError):
        temp_memory.update_part("TEST-0001-A", {"stock_qty": "zero"})


def test_update_numeric_string_coerced(temp_memory):
    # "5" as a string should be coerced to int 5 because existing field is int
    temp_memory.update_part("TEST-0001-A", {"stock_qty": "5"})
    result = temp_memory.get_by_sku("TEST-0001-A")
    assert result["stock_qty"] == 5


# ---------------------------------------------------------------------------
# remove_part
# ---------------------------------------------------------------------------

def test_remove_existing_part(temp_memory):
    temp_memory.remove_part("TEST-0001-A")
    assert temp_memory.get_by_sku("TEST-0001-A") is None


def test_remove_nonexistent_sku_raises(temp_memory):
    with pytest.raises(KeyError, match="not found"):
        temp_memory.remove_part("FAKE-0000-Z")


# ---------------------------------------------------------------------------
# list_by_category
# ---------------------------------------------------------------------------

def test_list_by_category_returns_correct_parts(temp_memory):
    results = temp_memory.list_by_category("hydraulic fittings")
    assert len(results) == 1
    assert results[0]["sku"] == "TEST-0001-A"


def test_list_by_category_empty(temp_memory):
    results = temp_memory.list_by_category("nonexistent category")
    assert results == []
