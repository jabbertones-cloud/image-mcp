"""Progressive tool discovery contract adapted from dcc-mcp group activation."""
import asyncio

from server import image_tools_server as srv


def _names():
    return {tool.name for tool in asyncio.run(srv.mcp.list_tools())}


def setup_function():
    srv._ACTIVE_TOOL_GROUPS.clear()


def test_default_surface_is_compact_but_keeps_control_tools():
    names = _names()
    assert "open_canvas" in names
    assert "search_tools" in names
    assert "activate_tool_group" in names
    assert "draw_line" not in names
    assert "sd_generate" not in names
    assert len(names) < 60


def test_search_finds_hidden_tool_and_reports_group():
    result = srv.search_tools("draw line")
    hit = next(item for item in result["results"] if item["name"] == "draw_line")
    assert hit["group"] == "drawing"


def test_activate_and_deactivate_group_changes_tools_list():
    assert "draw_line" not in _names()
    activated = srv.activate_tool_group("drawing")
    assert activated["changed"] is True
    assert "draw_line" in _names()
    deactivated = srv.deactivate_tool_group("drawing")
    assert deactivated["changed"] is True
    assert "draw_line" not in _names()


def test_all_registered_tools_remain_in_manager_when_hidden():
    visible = _names()
    registered = {tool.name for tool in srv.mcp._tool_manager.list_tools()}
    assert len(registered) > len(visible)
    assert "sd_generate" in registered
    assert "sd_generate" not in visible


def test_progressive_mode_can_be_disabled(monkeypatch):
    monkeypatch.setenv("IMAGETOOLS_PROGRESSIVE_TOOLS", "0")
    visible = _names()
    registered = {tool.name for tool in srv.mcp._tool_manager.list_tools()}
    assert visible == registered
