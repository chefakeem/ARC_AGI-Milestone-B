# ArcAgent.py — DSL + Search Tree agent (assignment-compliant)
# - Deterministic; NumPy + typing only
# - No test-output access; no problem_name use
# - ≤ 3 predictions (deduped)
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np

# -------------------- utils --------------------
def grids_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)

def most_freq(arr: np.ndarray) -> int:
    if arr.size == 0: return 0
    v, c = np.unique(arr, return_counts=True)
    return int(v[c.argmax()])

def dihedral_names() -> List[str]:
    return ["R0","R90","R180","R270","R0_M","R90_M","R180_M","R270_M"]

def apply_dihedral(arr: np.ndarray, name: str) -> np.ndarray:
    base = {"R0": arr, "R90": np.rot90(arr,1), "R180": np.rot90(arr,2), "R270": np.rot90(arr,3)}[name.split("_")[0]]
    return np.fliplr(base) if name.endswith("_M") else base

def cmap_from_xy(x: np.ndarray, y: np.ndarray) -> Optional[Dict[int,int]]:
    if x.shape != y.shape: return None
    m: Dict[int,int] = {}
    for c in np.unique(x):
        mask = (x == c)
        dv, dc = np.unique(y[mask], return_counts=True)
        if dv.size == 0: continue
        out = int(dv[dc.argmax()])
        if c in m and m[c] != out: return None
        m[c] = out
    return m

def apply_cmap(arr: np.ndarray, cmap: Dict[int,int]) -> np.ndarray:
    table = np.arange(10, dtype=arr.dtype)
    for k,v in cmap.items():
        if 0 <= k < 10 and 0 <= v < 10: table[k] = v
    return table[arr]

def compress_left(a: np.ndarray, bg:int) -> np.ndarray:
    H,W = a.shape
    out = np.full_like(a, bg)
    for r in range(H):
        row = [int(v) for v in a[r] if v != bg]
        if row: out[r, :len(row)] = row
    return out

def compress_up(a: np.ndarray, bg:int) -> np.ndarray:
    H,W = a.shape
    out = np.full_like(a, bg)
    for c in range(W):
        col = [int(a[r,c]) for r in range(H) if a[r,c] != bg]
        for r,v in enumerate(col): out[r,c] = v
    return out

def mirror_complete(a: np.ndarray, axis:str) -> np.ndarray:
    H,W = a.shape
    out = a.copy()
    if axis=="H":
        for r in range(H):
            rr = H-1-r
            take = np.where(out[r]!=0, out[r], out[rr])
            out[r]=take; out[rr]=take
    else:
        for c in range(W):
            cc = W-1-c
            col = out[:,c]; rcol = out[:,cc]
            take = np.where(col!=0, col, rcol)
            out[:,c]=take; out[:,cc]=take
    return out

def period2d(arr: np.ndarray) -> Tuple[int,int]:
    H,W = arr.shape
    def per1(v):
        n=v.size
        for p in range(1,n+1):
            if n%p==0 and np.all(v==np.resize(v[:p], n)): return p
        return n
    pH = min(per1(arr[:,c]) for c in range(W)) if W>0 else H
    pW = min(per1(arr[r,:]) for r in range(H)) if H>0 else W
    return max(1,pH), max(1,pW)

def tile(cell: np.ndarray, out_shape: Tuple[int,int]) -> np.ndarray:
    h,w = cell.shape; H,W = out_shape
    repH = (H+h-1)//h; repW=(W+w-1)//w
    big = np.tile(cell,(repH,repW))
    return big[:H,:W]

# --------------- components (4-neigh) ---------------
def cc(mask: np.ndarray):
    H,W = mask.shape; vis = np.zeros((H,W), bool); comps=[]
    for y in range(H):
        for x in range(W):
            if mask[y,x] and not vis[y,x]:
                q=[(y,x)]; vis[y,x]=True; pts=[(y,x)]
                while q:
                    cy,cx = q.pop()
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=cy+dy, cx+dx
                        if 0<=ny<H and 0<=nx<W and mask[ny,nx] and not vis[ny,nx]:
                            vis[ny,nx]=True; q.append((ny,nx)); pts.append((ny,nx))
                m = np.zeros_like(mask,bool)
                for yy,xx in pts: m[yy,xx]=True
                comps.append(m)
    return comps

def bbox(m: np.ndarray):
    ys,xs = np.where(m)
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())

# -------------------- DSL ops --------------------
# Each op is a function: f(grid) -> grid
# Parameterized ops are generated with a closure.

def op_dihedral(name:str) -> Callable[[np.ndarray], np.ndarray]:
    return lambda g: apply_dihedral(g, name)

def op_cmap_from_pair(x: np.ndarray, y: np.ndarray) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    cm = cmap_from_xy(x,y)
    if cm is None: return None
    return lambda g: apply_cmap(g, cm)

def op_compress_left(bg:int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda g: compress_left(g, bg)

def op_compress_up(bg:int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda g: compress_up(g, bg)

def op_sym(axis:str) -> Callable[[np.ndarray], np.ndarray]:
    return lambda g: mirror_complete(g, axis)

def op_translate(dy:int, dx:int, bg:int=0) -> Callable[[np.ndarray], np.ndarray]:
    def f(g: np.ndarray):
        H,W = g.shape; out = np.full_like(g, bg)
        ys0 = max(0, dy); ys1 = H+min(0, dy)
        xs0 = max(0, dx); xs1 = W+min(0, dx)
        yd0 = max(0,-dy); yd1 = yd0+(ys1-ys0)
        xd0 = max(0,-dx); xd1 = xd0+(xs1-xs0)
        if ys1>ys0 and xs1>xs0:
            out[ys0:ys1, xs0:xs1] = g[yd0:yd1, xd0:xd1]
        return out
    return f

def op_object_repaint(color:int) -> Callable[[np.ndarray], np.ndarray]:
    # recolor largest connected nonzero component uniformly
    def f(g: np.ndarray):
        comps = cc(g!=0)
        if not comps: return g.copy()
        comps.sort(key=lambda m: m.sum(), reverse=True)
        m = comps[0]
        out = g.copy()
        out[m] = color
        return out
    return f

def op_unit_cell_from_y(y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    pH,pW = period2d(y)
    cell = y[:pH,:pW]
    return lambda g: tile(cell, g.shape)

# ---------------- sequence scoring ----------------
def pixel_loss(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape: return 10**9
    return int(np.sum(a!=b))

def apply_seq(g: np.ndarray, ops: List[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    out = g
    for f in ops:
        out = f(out)
    return out

def sequence_fits_all(train_pairs: List[Tuple[np.ndarray,np.ndarray]], seq: List[Callable[[np.ndarray], np.ndarray]]) -> Tuple[int,int]:
    """Return (total_loss, failures). Perfect fit => (0,0)."""
    tot=0; fails=0
    for x,y in train_pairs:
        pred = apply_seq(x, seq)
        L = pixel_loss(pred, y); tot += L
        if L>0: fails += 1
    return tot, fails

# ---------------- ArcAgent (DSL + beam search) ----------------
class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem) -> List[np.ndarray]:
        # Gather training pairs
        train_pairs: List[Tuple[np.ndarray,np.ndarray]] = []
        for s in arc_problem.training_set():
            x = s.get_input_data().data()
            y = s.get_output_data().data()
            train_pairs.append((x,y))
        test_in = arc_problem.test_set().get_input_data().data()

        # 1) Build primitive op candidates from training evidence
        bg_out = most_freq(np.concatenate([y.ravel() for _x,y in train_pairs]))
        ops0: List[Callable[[np.ndarray], np.ndarray]] = []

        # dihedral (always available)
        for name in dihedral_names():
            ops0.append(op_dihedral(name))
        # symmetry completion hypotheses (check quickly on pairs)
        for axis in ("H","V"):
            if all(grids_equal(mirror_complete(x,axis), y) for x,y in train_pairs):
                ops0.append(op_sym(axis))
        # compression if exact on all pairs
        if all(grids_equal(compress_left(x,bg_out), y) for x,y in train_pairs):
            ops0.append(op_compress_left(bg_out))
        if all(grids_equal(compress_up(x,bg_out), y) for x,y in train_pairs):
            ops0.append(op_compress_up(bg_out))
        # color map learned consistently across all pairs
        global_map: Optional[Dict[int,int]] = None
        ok=True
        for x,y in train_pairs:
            cm = cmap_from_xy(x,y)
            if cm is None: ok=False; break
            if global_map is None: global_map = dict(cm)
            else:
                for k in set(global_map.keys()).intersection(cm.keys()):
                    if global_map[k]!=cm[k]: ok=False; break
                for k,v in cm.items():
                    if k not in global_map: global_map[k]=v
            if not ok: break
        if ok and global_map is not None:
            ops0.append(lambda g, cm=global_map: apply_cmap(g, cm))

        # translate small offsets
        for dy in (-2,-1,1,2):
            for dx in (-2,-1,1,2):
                ops0.append(op_translate(dy, dx, bg=0))
        # object repaint to frequent output color
        dom_out = most_freq(np.concatenate([y[y!=0] for _x,y in train_pairs if np.any(y!=0)])) if any(np.any(y!=0) for _x,y in train_pairs) else 1
        ops0.append(op_object_repaint(dom_out))
        # unit-cell tiling from any y if strictly periodic
        p_ok=True
        pHs=[]; pWs=[]
        for _x,y in train_pairs:
            ph,pw = period2d(y); pHs.append(ph); pWs.append(pw)
            if not grids_equal(tile(y[:ph,:pw], y.shape), y): p_ok=False
        if p_ok:
            # take cell from first y
            ops0.append(op_unit_cell_from_y(train_pairs[0][1]))

        # 2) Beam search over op sequences
        BeamItem = Tuple[int, List[Callable[[np.ndarray], np.ndarray]]]  # (loss, seq)
        beam: List[BeamItem] = [(0, [])]

        MAX_DEPTH = 3  # bump to 4 if runtime allows
        BEAM_WIDTH = 12

        best_exact: List[List[Callable[[np.ndarray], np.ndarray]]] = []
        best_approx: List[BeamItem] = []

        def update_best(seq):
            L, fails = sequence_fits_all(train_pairs, seq)
            if L==0 and fails==0:
                best_exact.append(seq)
            else:
                best_approx.append((L, seq))

        for depth in range(1, MAX_DEPTH+1):
            new_beam: List[BeamItem] = []
            for _cur_loss, seq in beam:
                for op in ops0:
                    seq2 = seq + [op]
                    L, _fails = sequence_fits_all(train_pairs, seq2)
                    new_beam.append((L, seq2))
            # keep top few for next expansion
            new_beam.sort(key=lambda t: t[0])
            beam = new_beam[:BEAM_WIDTH]
            # record promising ones
            for L, seq in beam:
                update_best(seq)

        # 3) Choose up to 3 sequences to produce predictions
        preds: List[np.ndarray] = []
        used_hashes: set = set()

        def add_pred(g: np.ndarray):
            key = (g.shape, g.tobytes())
            if key in used_hashes: return
            used_hashes.add(key)
            if len(preds) < 3:
                preds.append(g)

        # prefer exact fits first
        for seq in best_exact[:3]:
            add_pred(apply_seq(test_in, seq))
            if len(preds) >= 3: break
        # then best approximate
        if len(preds) < 3:
            best_approx.sort(key=lambda t: t[0])
            for L, seq in best_approx[:3]:
                add_pred(apply_seq(test_in, seq))
                if len(preds) >= 3: break
        # if still empty, identity fallback(s)
        if not preds:
            preds = [test_in.copy()]

        return preds
