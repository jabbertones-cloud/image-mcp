"""Progressive tool discovery without unregistering executable tools.

All implementation tools stay registered so existing callers keep working.
Only tools/list is projected to a compact core plus explicitly activated
groups. This mirrors the proven dcc-mcp-core separation of registry from
catalog exposure.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from threading import RLock
from typing import Iterable

CORE = {
    "new_canvas", "open_canvas", "save_canvas", "close_canvas",
    "list_canvases", "get_canvas_info", "get_canvas_preview", "inspect_region",
    "undo", "redo", "list_layers", "get_layer_preview",
    "discover_tools", "activate_tool_group", "deactivate_tool_group",
    "list_tool_groups",
}

PREFIX_GROUPS = {
    "stable_diffusion": ("sd_",),
    "qwen": ("qwen_",),
    "llada": ("llada_",),
    "sam": ("sam_", "sam1_"),
    "yolo": ("yolo_",),
    "background": ("birefnet_", "clipseg_"),
    "faces": ("face_",),
    "remote": ("remote_",),
}
EXACT_GROUPS = {
    "draw": {"draw_pixel","draw_line","draw_rectangle","draw_ellipse","draw_polygon","draw_arc","draw_text","draw_brush","eraser","flood_fill","pick_color"},
    "layers": {"add_layer","remove_layer","duplicate_layer","rename_layer","reorder_layer","set_active_layer","set_layer_visibility","set_layer_opacity","set_layer_blend_mode","set_layer_offset","merge_down","flatten_canvas","add_layer_mask","delete_layer_mask","apply_layer_mask","invert_layer_mask","fill_layer_mask","set_layer_mask_from_canvas"},
    "transform": {"crop","resize","rotate","flip","copy_region","paste_canvas","clear_region","warp_perspective","warp_mesh","liquify","distort","displace_by_map"},
    "files": {"image_info","convert_image","batch_convert","supported_formats","edit_image","resize_image","thumbnail_image","crop_image","rotate_image","flip_image","grayscale_image","extract_frames","build_animation","build_ico","split_ico","pdf_to_images","images_to_pdf","convert_mode","strip_metadata","copy_metadata"},
    "adjust": {"apply_filter","adjust","invert","grayscale","posterize","add_border","hue_saturation","levels","curves","color_balance","threshold","vibrance","channel_mixer","gradient_map","auto_levels","auto_contrast","equalize","gradient_fill","extract_channel","merge_channels","split_channels_to_layers"},
    "effects": {"clone_stamp","dodge_brush","burn_brush","blur_brush","sharpen_brush","add_drop_shadow","add_outer_glow","add_layer_stroke","motion_blur","radial_blur","lens_blur","tilt_shift","box_blur","add_watermark","make_qr_code","color_replace","rounded_corners","letterbox","white_balance","annotate","glitch_effect","auto_crop_to_content","smart_crop_to_aspect","pixelate","extract_palette","color_quantize","unsharp_mask","high_pass","add_vignette","add_noise","bilateral_filter","magic_wand","blend_canvases"},
    "analysis": {"perceptual_hash","hamming_distance","compare_images","diff_image","histogram"},
    "patterns": {"define_pattern","define_pattern_from_file","list_patterns","delete_pattern","fill_pattern","pattern_stamp","pattern_overlay","make_seamless"},
    "config": {"get_config","set_config"},
}


def group_for(name: str) -> str:
    for group, prefixes in PREFIX_GROUPS.items():
        if name.startswith(prefixes):
            return group
    for group, names in EXACT_GROUPS.items():
        if name in names:
            return group
    return "misc"


class ToolCatalog:
    def __init__(self, state_path: str | Path | None = None):
        raw = state_path or os.environ.get("IMAGETOOLS_TOOL_STATE")
        self.path = Path(raw).expanduser() if raw else None
        self.active: set[str] = set()
        self._lock = RLock()
        self._load()

    def _load(self):
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.active = {str(x) for x in data.get("activeGroups", [])}
        except (OSError, ValueError, TypeError):
            self.active = set()

    def _save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": 1, "activeGroups": sorted(self.active)}))
        os.replace(tmp, self.path)

    def activate(self, group: str, known_groups: Iterable[str]) -> bool:
        if group not in set(known_groups):
            raise ValueError(f"unknown tool group: {group}")
        with self._lock:
            changed = group not in self.active
            self.active.add(group); self._save()
            return changed

    def deactivate(self, group: str) -> bool:
        with self._lock:
            changed = group in self.active
            self.active.discard(group); self._save()
            return changed

    def exposed(self, names: Iterable[str]) -> set[str]:
        names = set(names)
        return {n for n in names if n in CORE or group_for(n) in self.active}

    def search(self, names: Iterable[str], query: str, limit: int = 20):
        q = query.strip().lower()
        rows = []
        for name in names:
            group = group_for(name)
            if not q or q in name.lower() or q in group:
                rows.append({"name": name, "group": group, "active": name in CORE or group in self.active})
        rows.sort(key=lambda x: (not x["active"], x["group"], x["name"]))
        return rows[:max(1, min(int(limit), 50))]
