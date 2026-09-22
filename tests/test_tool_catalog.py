from server.tool_catalog import ToolCatalog, group_for


def test_compact_default_exposes_core_not_heavy_tools():
    c = ToolCatalog()
    names = {"new_canvas","open_canvas","sd_generate","qwen_edit_image","draw_line","discover_tools"}
    assert c.exposed(names) == {"new_canvas","open_canvas","discover_tools"}


def test_activate_and_deactivate_group():
    c = ToolCatalog()
    names = {"new_canvas","draw_line","draw_text","sd_generate"}
    assert c.activate("draw", {"draw","stable_diffusion"})
    assert c.exposed(names) == {"new_canvas","draw_line","draw_text"}
    assert c.deactivate("draw")
    assert c.exposed(names) == {"new_canvas"}


def test_state_replays_after_restart(tmp_path):
    p = tmp_path / "groups.json"
    c1 = ToolCatalog(p)
    c1.activate("analysis", {"analysis"})
    c2 = ToolCatalog(p)
    assert "analysis" in c2.active
    assert "compare_images" in c2.exposed({"compare_images"})


def test_search_is_bounded_and_reports_group():
    c = ToolCatalog()
    rows = c.search(["sd_generate","sd_load","draw_line"], "sd", limit=1)
    assert len(rows) == 1
    assert rows[0]["group"] == "stable_diffusion"


def test_prefix_classification():
    assert group_for("qwen_edit_image") == "qwen"
    assert group_for("sam1_segment_box") == "sam"
    assert group_for("remote_queue_status") == "remote"

def test_real_server_default_list_is_small_and_activation_expands(monkeypatch, tmp_path):
    import asyncio
    from server import image_tools_server as srv
    srv._tool_catalog.path = tmp_path / "state.json"
    srv._tool_catalog.active.clear()
    base = asyncio.run(srv.mcp.list_tools())
    base_names = {t.name for t in base}
    assert "new_canvas" in base_names
    assert "discover_tools" in base_names
    assert "qwen_edit_image" not in base_names
    assert len(base_names) <= 20

    groups = {group_for(t.name) for t in srv.mcp._tool_manager.list_tools()}
    srv._tool_catalog.activate("qwen", groups)
    expanded = asyncio.run(srv.mcp.list_tools())
    expanded_names = {t.name for t in expanded}
    assert "qwen_edit_image" in expanded_names
    assert len(expanded_names) > len(base_names)
