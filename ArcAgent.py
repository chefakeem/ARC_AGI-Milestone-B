import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional, Set, Callable

# =========================================================
# Shared helpers
# =========================================================

def color_counts(arr: np.ndarray):
    vals, counts = np.unique(arr, return_counts=True)
    return dict(zip(vals, counts))

def grid_score(pred: Optional[np.ndarray], target: np.ndarray) -> float:
    if pred is None or pred.shape != target.shape:
        return 0.0
    return float(np.mean(pred == target))

def count_components_by_color(arr: np.ndarray, ignore_colors=None):
    """Return {color: number_of_components} using 4-connectivity."""
    if ignore_colors is None:
        ignore_colors = set()
    else:
        ignore_colors = set(ignore_colors)

    h, w = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    comp_counts = defaultdict(int)

    for r in range(h):
        for c in range(w):
            val = arr[r, c]
            if val in ignore_colors or visited[r, c]:
                continue

            comp_counts[val] += 1
            q = deque([(r, c)])
            visited[r, c] = True

            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = rr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w and
                        not visited[nr, nc] and arr[nr, nc] == val):
                        visited[nr, nc] = True
                        q.append((nr, nc))

    return dict(comp_counts)

def connected_components(arr: np.ndarray, colors: Set[int]) -> List[List[Tuple[int,int]]]:
    """Components for pixels whose value is in colors (4-connectivity)."""
    h, w = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not visited[r,c] and arr[r,c] in colors:
                col = arr[r,c]
                q=deque([(r,c)])
                visited[r,c]=True
                cells=[]
                while q:
                    rr,cc=q.popleft()
                    cells.append((rr,cc))
                    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr,nc=rr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and arr[nr,nc]==col:
                            visited[nr,nc]=True
                            q.append((nr,nc))
                comps.append(cells)
    return comps

def bbox(cells: List[Tuple[int,int]]) -> Tuple[int,int,int,int]:
    rs=[r for r,_ in cells]
    cs=[c for _,c in cells]
    return min(rs), max(rs), min(cs), max(cs)

def dihedral_transforms(mask: np.ndarray) -> List[np.ndarray]:
    """All 8 rotations/flips of a binary mask (unique)."""
    outs=[]
    for k in range(4):
        rot=np.rot90(mask,k)
        outs.append(rot)
        outs.append(np.fliplr(rot))
    uniq=[]
    for m in outs:
        if not any(np.array_equal(m,u) for u in uniq):
            uniq.append(m)
    return uniq


# =========================================================
# Preprocessing / noise removal
# =========================================================

def detect_background(x: np.ndarray) -> int:
    vals, counts = np.unique(x, return_counts=True)
    return int(vals[np.argmax(counts)])

def remove_background(x: np.ndarray):
    bg = detect_background(x)
    y = x.copy()
    y[y == bg] = 0
    return y, bg

def strip_grid_lines(x: np.ndarray):
    """
    Remove dominant nonzero color ONLY if it forms full separator rows/cols.
    """
    y = x.copy()
    h, w = y.shape
    nz = y[y != 0]
    if nz.size == 0:
        return y, None

    vals, counts = np.unique(nz, return_counts=True)
    line_color = int(vals[np.argmax(counts)])

    full_rows = [r for r in range(h) if np.all(y[r, :] == line_color)]
    full_cols = [c for c in range(w) if np.all(y[:, c] == line_color)]

    if len(full_rows) + len(full_cols) == 0:
        return y, None

    y[y == line_color] = 0
    return y, line_color

def filter_tiny_components(x: np.ndarray, min_size: Optional[int]=None):
    """
    Drop tiny components (noise).
    min_size dynamic if None.
    """
    h, w = x.shape
    if min_size is None:
        min_size = max(2, (h*w)//200)

    y = x.copy()
    visited = np.zeros_like(x, dtype=bool)

    for r in range(h):
        for c in range(w):
            if x[r,c] != 0 and not visited[r,c]:
                color = x[r,c]
                q=deque([(r,c)])
                visited[r,c]=True
                cells=[]
                while q:
                    rr,cc=q.popleft()
                    cells.append((rr,cc))
                    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr,nc=rr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and x[nr,nc]==color:
                            visited[nr,nc]=True
                            q.append((nr,nc))
                if len(cells) < min_size:
                    for rr,cc in cells:
                        y[rr,cc] = 0
    return y

def make_views(x: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Produce multiple cleaned versions of x.
    Solvers can be scored on each view.
    """
    views = [("raw", x)]

    gx, _ = strip_grid_lines(x)
    if not np.array_equal(gx, x):
        views.append(("grid_stripped", gx))

    tx = filter_tiny_components(x)
    if not np.array_equal(tx, x):
        views.append(("tiny_filtered", tx))

    bx, _ = remove_background(x)
    if not np.array_equal(bx, x):
        views.append(("bg_removed", bx))

    return views


# =========================================================
# Heuristic 1: component summary (81-family)
# =========================================================

def heuristic_component_summary(x: np.ndarray) -> np.ndarray:
    counts = color_counts(x)
    nonzero = {k:v for k,v in counts.items() if k!=0}
    if not nonzero:
        return np.zeros((1,1), dtype=int)

    grid_color = max(nonzero, key=nonzero.get)

    comps = count_components_by_color(x, ignore_colors={0, grid_color})
    if not comps:
        return np.zeros((1,1), dtype=int)

    items = sorted(comps.items(), key=lambda kv:(kv[1], kv[0]))
    max_len = max(c for _,c in items)

    out = np.zeros((len(items), max_len), dtype=int)
    for r,(color,cnt) in enumerate(items):
        out[r,:cnt] = color
    return out


# =========================================================
# Heuristic 2: shape packing into slots (67-family)
# =========================================================

def detect_ground_color(x: np.ndarray) -> int:
    h,w = x.shape
    bottom = x[max(0,h-2):h,:]
    vals, counts = np.unique(bottom[bottom!=0], return_counts=True)
    if len(vals)==0:
        vals2, counts2 = np.unique(x[x!=0], return_counts=True)
        return int(vals2[np.argmax(counts2)]) if len(vals2)>0 else 0
    return int(vals[np.argmax(counts)])

def get_shapes_above_ground(x: np.ndarray, ground_color: int) -> List[Dict]:
    colors = set(np.unique(x)) - {0, ground_color}
    comps = connected_components(x, colors)
    shapes=[]
    for cells in comps:
        col = x[cells[0]]
        r0,r1,c0,c1 = bbox(cells)
        grid = np.zeros((r1-r0+1, c1-c0+1), dtype=int)
        for r,c in cells:
            grid[r-r0, c-c0] = col
        shapes.append({
            "color": int(col),
            "bbox": (r0,r1,c0,c1),
            "mask": (grid!=0).astype(int)
        })
    shapes.sort(key=lambda s: (s["bbox"][2], s["bbox"][0]))
    return shapes

def get_slot_mask(x: np.ndarray, ground_color: int) -> np.ndarray:
    h,w = x.shape
    slot = np.zeros_like(x, dtype=bool)
    slot |= (x == ground_color)
    for r in range(h-1):
        above = (x[r,:] != 0)
        below_ground = (x[r+1,:] == ground_color)
        slot[r,:] |= (above & below_ground)
    return slot

def slot_components(slot_mask: np.ndarray) -> List[List[Tuple[int,int]]]:
    h,w = slot_mask.shape
    visited=np.zeros_like(slot_mask, dtype=bool)
    comps=[]
    for r in range(h):
        for c in range(w):
            if slot_mask[r,c] and not visited[r,c]:
                q=deque([(r,c)])
                visited[r,c]=True
                cells=[]
                while q:
                    rr,cc=q.popleft()
                    cells.append((rr,cc))
                    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr,nc=rr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and slot_mask[nr,nc] and not visited[nr,nc]:
                            visited[nr,nc]=True
                            q.append((nr,nc))
                comps.append(cells)
    comps.sort(key=lambda cells: min(c for _,c in cells))
    return comps

def place_shape_on_slot(slot_cells, shape_mask):
    srs=[r for r,_ in slot_cells]
    scs=[c for _,c in slot_cells]
    rmin,rmax,cmin,cmax=min(srs),max(srs),min(scs),max(scs)
    sh, sw = shape_mask.shape

    if sh != (rmax-rmin+1) or sw != (cmax-cmin+1):
        return None

    slot_bbox_mask = np.zeros((sh,sw), dtype=int)
    for r,c in slot_cells:
        slot_bbox_mask[r-rmin, c-cmin] = 1

    if np.array_equal(slot_bbox_mask, shape_mask):
        return (rmin, cmin)
    return None

def backtrack_assign(shapes, slots, x, ground_color):
    used=[False]*len(slots)
    placements=[]

    def rec(i):
        if i==len(shapes):
            out = np.zeros_like(x)
            out[x==ground_color]=ground_color
            for shape_idx, slot_idx, (r0,c0), mask in placements:
                color=shapes[shape_idx]["color"]
                sh,sw=mask.shape
                for rr in range(sh):
                    for cc in range(sw):
                        if mask[rr,cc]:
                            out[r0+rr, c0+cc]=color
            return out

        base_mask=shapes[i]["mask"]
        for tmask in dihedral_transforms(base_mask):
            for j,slot_cells in enumerate(slots):
                if used[j]:
                    continue
                top_left = place_shape_on_slot(slot_cells, tmask)
                if top_left is None:
                    continue
                used[j]=True
                placements.append((i,j,top_left,tmask))
                res=rec(i+1)
                if res is not None:
                    return res
                placements.pop()
                used[j]=False
        return None

    return rec(0)

def heuristic_shape_pack_slots(x: np.ndarray):
    ground_color = detect_ground_color(x)
    shapes = get_shapes_above_ground(x, ground_color)
    slots = slot_components(get_slot_mask(x, ground_color))
    if not shapes or not slots:
        return None
    return backtrack_assign(shapes, slots, x, ground_color)


# =========================================================
# Heuristic 3: grid tile template completion (2546-family)
# =========================================================

def find_full_line_indices(arr, line_color):
    h,w = arr.shape
    horiz = [r for r in range(h) if np.all(arr[r,:]==line_color)]
    vert  = [c for c in range(w) if np.all(arr[:,c]==line_color)]
    return horiz, vert

def tile_ranges(size, separators):
    seps = [-1] + sorted(separators) + [size-1]
    ranges=[]
    for i in range(len(seps)-1):
        start = seps[i]+1
        end   = seps[i+1]-1
        if start<=end:
            ranges.append((start,end))
    return ranges

def extract_tiles(arr, line_color):
    hr, vc = find_full_line_indices(arr, line_color)
    rr = tile_ranges(arr.shape[0], hr)
    cc = tile_ranges(arr.shape[1], vc)
    tiles=[]
    for r0,r1 in rr:
        for c0,c1 in cc:
            tiles.append(((r0,r1,c0,c1), arr[r0:r1+1, c0:c1+1]))
    return tiles

def dihedral_grids(grid):
    outs=[]
    for k in range(4):
        rot=np.rot90(grid,k)
        outs.append(rot)
        outs.append(np.fliplr(rot))
    uniq=[]
    for g in outs:
        if not any(np.array_equal(g,u) for u in uniq):
            uniq.append(g)
    return uniq

def heuristic_tile_template_completion(x: np.ndarray) -> Optional[np.ndarray]:
    nz = x[x != 0]
    if nz.size == 0:
        return None
    vals, counts = np.unique(nz, return_counts=True)
    line_color = int(vals[np.argmax(counts)])

    hr, vc = find_full_line_indices(x, line_color)
    if len(hr) + len(vc) == 0:
        return None

    tiles = extract_tiles(x, line_color)
    if not tiles:
        return None

    def motif_mask(tile):
        return (tile != 0) & (tile != line_color)

    tile_infos=[]
    for bbox_, tile in tiles:
        mm = motif_mask(tile)
        tile_infos.append((bbox_, tile, mm, int(mm.sum())))

    template_bbox, template_tile, template_mm, _ = max(tile_infos, key=lambda t: t[3])
    if template_mm.sum() == 0:
        return None

    template_colors = sorted(set(np.unique(template_tile)) - {0, line_color})
    out = x.copy()

    template_masks_by_color = {}
    for col in template_colors:
        base_mask = (template_tile == col).astype(int)
        template_masks_by_color[col] = dihedral_grids(base_mask)

    for bbox_, tile, mm, _ in tile_infos:
        r0,r1,c0,c1 = bbox_
        tile_out = tile.copy()

        for col in template_colors:
            current = (tile == col).astype(int)
            if current.sum() == 0:
                continue

            best_t = None
            for tmask in template_masks_by_color[col]:
                # IMPORTANT: only compare same-sized masks
                if tmask.shape != current.shape:
                    continue

                if np.all((current == 1) <= (tmask == 1)):
                    best_t = tmask
                    break

            if best_t is not None:
                fill_locations = (best_t == 1) & (current == 0)
                tile_out[fill_locations] = col

        out[r0:r1+1, c0:c1+1] = tile_out

    return out



# =========================================================
# Scoring + selection (THRESH=0.80)
# =========================================================

THRESH = 0.80
MAX_PREDS = 3

def score_heuristic_on_training(problem, fn: Callable[[np.ndarray], Optional[np.ndarray]]) -> float:
    """
    For each training pair: try heuristic on multiple views; keep best score.
    Mean over training pairs is returned.
    """
    scores=[]
    for s in problem.training_set():
        x = s.get_input_data().data()
        y = s.get_output_data().data()

        best = 0.0
        for _, view in make_views(x):
            pred = fn(view)
            best = max(best, grid_score(pred, y))
        scores.append(best)

    return float(np.mean(scores)) if scores else 0.0

def choose_predictions(problem) -> List[np.ndarray]:
    """
    Define heuristics locally (no global list), score them, pick best ≥ THRESH.
    """
    heuristics = [
        ("component_summary", heuristic_component_summary),
        ("shape_pack_slots", heuristic_shape_pack_slots),
        ("tile_template_completion", heuristic_tile_template_completion),
    ]

    ranked=[]
    for name, fn in heuristics:
        sc = score_heuristic_on_training(problem, fn)
        ranked.append((sc, name, fn))

    ranked.sort(reverse=True, key=lambda t: t[0])

    test_in = problem.test_set().get_input_data().data()
    preds=[]

    for sc, name, fn in ranked:
        if sc < THRESH:
            continue

        for _, v in make_views(test_in):
            pred = fn(v)
            if pred is None:
                continue
            if not any(np.array_equal(pred, p) for p in preds):
                preds.append(pred)
            if len(preds) >= MAX_PREDS:
                break
        if len(preds) >= MAX_PREDS:
            break

    return preds


# =========================================================
# ArcAgent
# =========================================================

class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem) -> List[np.ndarray]:
        preds = choose_predictions(arc_problem)
        if preds:
            return preds[:MAX_PREDS]

        # Baseline fallback so CSV always fills:
        test_in = arc_problem.test_set().get_input_data().data()
        return [test_in.copy()]
