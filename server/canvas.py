"""In-memory canvas store with layers + bounded undo/redo.

Each canvas is a fixed-size document containing a list of ``Layer`` objects
(bottom-to-top z-order) plus an ``active_index`` cursor. Drawing/filter ops
target the active layer; canvas-wide transforms (resize/crop/rotate/flip)
apply to all layers. ``compose()`` flattens the stack into a single RGBA
image for save/preview.

Single-layer canvases — what ``new_canvas`` / ``open_canvas`` produce by
default — behave exactly like the pre-layers store: there's just one
layer and it's always active.

Undo/redo snapshots the entire layer stack (and active index), so any
mutation can be rolled back atomically.
"""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from PIL import Image

from .layers import Layer, compose_layers

MAX_HISTORY = 32
# Byte budget for undo history across ALL open canvases. Snapshots deep-copy
# every layer, so 32 steps on a 4K-class canvas would otherwise pin gigabytes
# of RAM — and the real resource is process RAM, not any single canvas, so the
# cap is store-wide. When over budget the globally-oldest undo snapshot is
# evicted first, always leaving each canvas at least one undo step. Redo stacks
# are never byte-trimmed: a redo stack can only grow by moving snapshots off
# the (already-capped) undo stack, so it inherits the same bound, and trimming
# it from either end would silently discard reachable redo states.
MAX_HISTORY_MB = int(os.environ.get("IMAGETOOLS_UNDO_MAX_MB", "512"))


def _layers_cost(layers: list[Layer]) -> int:
    """Approximate resident bytes of a layer stack (RGBA + optional mask)."""
    total = 0
    for l in layers:
        total += l.image.width * l.image.height * 4
        if l.mask is not None:
            total += l.mask.width * l.mask.height
    return total


@dataclass
class CanvasState:
    """Snapshot-able state. Stored on the undo/redo stacks. ``cost`` is derived
    from ``layers`` and ``seq`` is a global capture order used to evict the
    oldest undo snapshot across canvases; callers never set them by hand."""
    layers: list[Layer]
    active_index: int
    cost: int = 0
    seq: int = 0

    def __post_init__(self) -> None:
        if not self.cost:
            self.cost = _layers_cost(self.layers)


@dataclass
class CanvasEntry:
    width: int
    height: int
    layers: list[Layer]
    active_index: int = 0
    path: str | None = None
    undo_stack: list[CanvasState] = field(default_factory=list)
    redo_stack: list[CanvasState] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def active_layer(self) -> Layer:
        return self.layers[self.active_index]


class CanvasStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, CanvasEntry] = {}
        self._seq = 0  # monotonic capture counter for global undo eviction

    # --------------------------------------------------------------- history
    def _capture(self, e: CanvasEntry) -> CanvasState:
        """Deep-copy the current layer stack into a history snapshot."""
        self._seq += 1
        return CanvasState(
            layers=[l.copy() for l in e.layers],
            active_index=e.active_index,
            seq=self._seq,
        )

    def _enforce_global_undo_budget(self) -> None:
        """Evict the globally-oldest undo snapshot until total undo bytes fit
        the store-wide budget, always keeping >=1 undo step per canvas."""
        budget = MAX_HISTORY_MB * 1024 * 1024
        total = sum(s.cost for e in self._items.values() for s in e.undo_stack)
        while total > budget:
            oldest = min(
                (e.undo_stack[0] for e in self._items.values() if len(e.undo_stack) > 1),
                key=lambda s: s.seq, default=None,
            )
            if oldest is None:  # every stack down to its last step
                break
            for e in self._items.values():
                if e.undo_stack and e.undo_stack[0] is oldest:
                    total -= e.undo_stack.pop(0).cost
                    break

    # --------------------------------------------------------------- lifecycle
    def put_image(self, image: Image.Image, *, canvas_id: str | None = None,
                  path: str | None = None, layer_name: str = "Background") -> str:
        """Create a single-layer canvas from a flat image. The original mode
        is preserved on save; layers are always stored RGBA internally."""
        rgba = image.convert("RGBA") if image.mode != "RGBA" else image.copy()
        layer = Layer(name=layer_name, image=rgba)
        return self.put_layers(layer.image.width, layer.image.height, [layer],
                               canvas_id=canvas_id, path=path)

    def put_layers(self, width: int, height: int, layers: list[Layer], *,
                   canvas_id: str | None = None, path: str | None = None) -> str:
        if not layers:
            raise ValueError("canvas must have at least one layer")
        with self._lock:
            cid = canvas_id or f"cv_{uuid.uuid4().hex[:8]}"
            self._items[cid] = CanvasEntry(
                width=int(width), height=int(height),
                layers=list(layers), active_index=len(layers) - 1, path=path,
            )
            return cid

    def has(self, canvas_id: str) -> bool:
        with self._lock:
            return canvas_id in self._items

    def entry(self, canvas_id: str) -> CanvasEntry:
        with self._lock:
            try:
                return self._items[canvas_id]
            except KeyError:
                raise KeyError(
                    f"unknown canvas_id {canvas_id!r}. Open one via "
                    f"new_canvas/open_canvas, or call list_canvases."
                )

    def close(self, canvas_id: str) -> None:
        with self._lock:
            self._items.pop(canvas_id, None)

    def list_ids(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._items.keys()))

    def summary(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "canvas_id": cid,
                    "width": e.width,
                    "height": e.height,
                    "n_layers": len(e.layers),
                    "active_index": e.active_index,
                    "path": e.path,
                    "undo_depth": len(e.undo_stack),
                    "redo_depth": len(e.redo_stack),
                }
                for cid, e in self._items.items()
            ]

    # --------------------------------------------------------------- read
    def compose(self, canvas_id: str) -> Image.Image:
        """Flatten visible layers into a single RGBA image."""
        e = self.entry(canvas_id)
        return compose_layers(e.layers, e.size)

    def compose_region(self, canvas_id: str,
                       box: tuple[int, int, int, int]) -> Image.Image:
        """Flatten visible layers over just ``box`` (x0, y0, x1, y1), composited
        into a region-sized buffer — cheap for zooming into a large canvas."""
        e = self.entry(canvas_id)
        return compose_layers(e.layers, e.size, region=box)

    def active_layer(self, canvas_id: str) -> Layer:
        return self.entry(canvas_id).active_layer

    def get_active_image(self, canvas_id: str) -> Image.Image:
        """Image of the active layer — the write target for drawing ops."""
        return self.entry(canvas_id).active_layer.image

    def replace_active_image(self, canvas_id: str, image: Image.Image) -> None:
        with self._lock:
            e = self.entry(canvas_id)
            e.active_layer.image = (
                image if image.mode == "RGBA" else image.convert("RGBA")
            )

    # --------------------------------------------------------------- layer ops
    def add_layer(self, canvas_id: str, *, name: str | None = None,
                  fill: tuple[int, int, int, int] = (0, 0, 0, 0),
                  above: int | None = None) -> int:
        """Insert a new transparent layer. Returns its index. ``above`` is
        the index to insert above (default: top of stack)."""
        with self._lock:
            e = self.entry(canvas_id)
            img = Image.new("RGBA", e.size, fill)
            layer = Layer(name=name or f"Layer {len(e.layers) + 1}", image=img)
            idx = len(e.layers) if above is None else int(above) + 1
            idx = max(0, min(idx, len(e.layers)))
            e.layers.insert(idx, layer)
            e.active_index = idx
            return idx

    def remove_layer(self, canvas_id: str, index: int) -> None:
        with self._lock:
            e = self.entry(canvas_id)
            if len(e.layers) <= 1:
                raise ValueError("cannot remove the last layer; flatten or close the canvas instead")
            if not (0 <= index < len(e.layers)):
                raise IndexError(f"layer index {index} out of range")
            e.layers.pop(index)
            if e.active_index >= len(e.layers):
                e.active_index = len(e.layers) - 1
            elif e.active_index > index:
                e.active_index -= 1

    def duplicate_layer(self, canvas_id: str, index: int) -> int:
        with self._lock:
            e = self.entry(canvas_id)
            if not (0 <= index < len(e.layers)):
                raise IndexError(f"layer index {index} out of range")
            dup = e.layers[index].copy()
            dup.name = e.layers[index].name + " copy"
            e.layers.insert(index + 1, dup)
            e.active_index = index + 1
            return index + 1

    def reorder_layer(self, canvas_id: str, src: int, dst: int) -> None:
        with self._lock:
            e = self.entry(canvas_id)
            n = len(e.layers)
            if not (0 <= src < n):
                raise IndexError(f"src index {src} out of range")
            dst = max(0, min(dst, n - 1))
            if src == dst:
                return
            layer = e.layers.pop(src)
            e.layers.insert(dst, layer)
            # Keep active_index pointing at the same layer.
            if e.active_index == src:
                e.active_index = dst
            elif src < e.active_index <= dst:
                e.active_index -= 1
            elif dst <= e.active_index < src:
                e.active_index += 1

    def set_active(self, canvas_id: str, index: int) -> None:
        with self._lock:
            e = self.entry(canvas_id)
            if not (0 <= index < len(e.layers)):
                raise IndexError(f"layer index {index} out of range")
            e.active_index = index

    def merge_down(self, canvas_id: str, index: int) -> None:
        """Composite layer ``index`` onto the layer below it, removing the top one.

        The result keeps the bottom layer's *name*; its image is replaced with
        a canvas-sized flat composite, so offset/opacity/blend_mode/mask are
        reset to neutral values (the new image already encodes those effects).
        """
        with self._lock:
            e = self.entry(canvas_id)
            if not (1 <= index < len(e.layers)):
                raise ValueError("merge_down: need a layer index >= 1 (one below it)")
            top = e.layers[index]
            bottom = e.layers[index - 1]
            merged_full = compose_layers([bottom, top], e.size)
            bottom.image = merged_full
            bottom.offset = (0, 0)
            bottom.opacity = 1.0
            bottom.blend_mode = "normal"
            bottom.mask = None
            e.layers.pop(index)
            if e.active_index >= len(e.layers):
                e.active_index = len(e.layers) - 1
            elif e.active_index >= index:
                e.active_index -= 1

    def flatten(self, canvas_id: str) -> None:
        """Collapse all visible layers into a single Background layer."""
        with self._lock:
            e = self.entry(canvas_id)
            flat = compose_layers(e.layers, e.size)
            e.layers = [Layer(name="Background", image=flat)]
            e.active_index = 0

    def snapshot(self, canvas_id: str) -> None:
        """Capture pre-mutation state. Clears redo (new branch)."""
        with self._lock:
            e = self.entry(canvas_id)
            e.undo_stack.append(self._capture(e))
            while len(e.undo_stack) > MAX_HISTORY:
                e.undo_stack.pop(0)
            e.redo_stack.clear()
            self._enforce_global_undo_budget()

    def undo(self, canvas_id: str) -> bool:
        with self._lock:
            e = self.entry(canvas_id)
            if not e.undo_stack:
                return False
            e.redo_stack.append(self._capture(e))
            prev = e.undo_stack.pop()
            e.layers = prev.layers
            e.active_index = prev.active_index
            return True

    def redo(self, canvas_id: str) -> bool:
        with self._lock:
            e = self.entry(canvas_id)
            if not e.redo_stack:
                return False
            e.undo_stack.append(self._capture(e))
            while len(e.undo_stack) > MAX_HISTORY:
                e.undo_stack.pop(0)
            nxt = e.redo_stack.pop()
            e.layers = nxt.layers
            e.active_index = nxt.active_index
            self._enforce_global_undo_budget()
            return True

    # --------------------------------------------------------------- canvas-wide
    def map_all_layers(self, canvas_id: str, fn) -> None:
        """Apply ``fn(image) -> image`` to every layer's image. Used by
        canvas-wide transforms (resize/rotate/flip). The function should
        return a new image; size changes are caller's responsibility."""
        with self._lock:
            e = self.entry(canvas_id)
            for layer in e.layers:
                layer.image = fn(layer.image)
                if layer.mask is not None:
                    layer.mask = fn(layer.mask)

    def set_canvas_size(self, canvas_id: str, width: int, height: int) -> None:
        with self._lock:
            e = self.entry(canvas_id)
            e.width = int(width)
            e.height = int(height)


# module-level singleton — MCP tools share one store
store = CanvasStore()
