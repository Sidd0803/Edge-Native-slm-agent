from unittest.mock import MagicMock, patch

from cli import handle_command


# ---------------------------------------------------------------------------
# /update
# ---------------------------------------------------------------------------

def test_update_valid_field_calls_memory(capsys):
    # Valid update reaches memory with correct args and prints confirmation
    memory = MagicMock()
    handle_command("/update HF-0001-A stock_qty 10", memory)
    memory.update_part.assert_called_once_with("HF-0001-A", {"stock_qty": 10})
    assert "Updated" in capsys.readouterr().out


def test_update_uppercases_sku():
    # SKU is normalised to uppercase before being passed to memory
    memory = MagicMock()
    handle_command("/update hf-0001-a stock_qty 10", memory)
    memory.update_part.assert_called_once_with("HF-0001-A", {"stock_qty": 10})


def test_update_unknown_sku_prints_error(capsys):
    # KeyError from memory (SKU not found) is caught and printed, not raised
    memory = MagicMock()
    memory.update_part.side_effect = KeyError("SKU FAKE-0000-Z not found")
    handle_command("/update FAKE-0000-Z stock_qty 5", memory)
    assert "Error" in capsys.readouterr().out


def test_update_unknown_field_prints_error(capsys):
    # KeyError from memory (bad field name) is caught and printed, not raised
    memory = MagicMock()
    memory.update_part.side_effect = KeyError("Unknown field 'qty' for SKU HF-0001-A")
    handle_command("/update HF-0001-A qty 5", memory)
    assert "Error" in capsys.readouterr().out


def test_update_type_mismatch_prints_error(capsys):
    # ValueError from memory (wrong value type) is caught and printed, not raised
    memory = MagicMock()
    memory.update_part.side_effect = ValueError("Field 'stock_qty' expects int, got 'zero'")
    handle_command("/update HF-0001-A stock_qty zero", memory)
    assert "Error" in capsys.readouterr().out


def test_update_coerces_integer_string():
    # "5" as a CLI string is coerced to int before reaching memory
    memory = MagicMock()
    handle_command("/update HF-0001-A stock_qty 5", memory)
    memory.update_part.assert_called_once_with("HF-0001-A", {"stock_qty": 5})


def test_update_coerces_float_string():
    # "9.99" as a CLI string is coerced to float before reaching memory
    memory = MagicMock()
    handle_command("/update HF-0001-A unit_price 9.99", memory)
    memory.update_part.assert_called_once_with("HF-0001-A", {"unit_price": 9.99})


# ---------------------------------------------------------------------------
# /remove
# ---------------------------------------------------------------------------

def test_remove_valid_sku_calls_memory(capsys):
    memory = MagicMock()
    handle_command("/remove HF-0001-A", memory)
    memory.remove_part.assert_called_once_with("HF-0001-A")
    assert "Removed" in capsys.readouterr().out


def test_remove_uppercases_sku():
    # SKU is normalised to uppercase before being passed to memory
    memory = MagicMock()
    handle_command("/remove hf-0001-a", memory)
    memory.remove_part.assert_called_once_with("HF-0001-A")


def test_remove_unknown_sku_prints_error(capsys):
    memory = MagicMock()
    memory.remove_part.side_effect = KeyError("SKU FAKE-0000-Z not found")
    handle_command("/remove FAKE-0000-Z", memory)
    assert "Error" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# /add
# ---------------------------------------------------------------------------

def test_add_success(capsys):
    memory = MagicMock()
    inputs = ["TEST-ADD-A", "Test part desc", "hydraulic fittings", "5", "10.0", "Supplier X", "3", "2"]
    with patch("builtins.input", side_effect=inputs):
        handle_command("/add", memory)
    memory.add_part.assert_called_once()
    call_kwargs = memory.add_part.call_args[0][0]
    assert call_kwargs["sku"] == "TEST-ADD-A"
    assert call_kwargs["stock_qty"] == 5
    assert "Added" in capsys.readouterr().out


def test_add_duplicate_sku_prints_error(capsys):
    memory = MagicMock()
    memory.add_part.side_effect = ValueError("SKU TEST-ADD-A already exists")
    inputs = ["TEST-ADD-A", "Desc", "hydraulic fittings", "5", "10.0", "Supplier", "3", "2"]
    with patch("builtins.input", side_effect=inputs):
        handle_command("/add", memory)
    assert "Error" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# /list
# ---------------------------------------------------------------------------

def test_list_category_prints_parts(capsys):
    memory = MagicMock()
    memory.list_by_category.return_value = [{
        "sku": "HF-0001-A",
        "description": "Hydraulic fitting",
        "stock_qty": 10,
        "unit_price": 12.5,
    }]
    handle_command("/list hydraulic fittings", memory)
    out = capsys.readouterr().out
    assert "HF-0001-A" in out


def test_list_empty_category_prints_message(capsys):
    memory = MagicMock()
    memory.list_by_category.return_value = []
    handle_command("/list nonexistent", memory)
    assert "No parts found" in capsys.readouterr().out
