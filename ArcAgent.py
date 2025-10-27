import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init. Be aware that it gets called only once
        and then the solve method will get called several times.
        """
        
        self.MAX_DEPTH = 2
        self.BEAM_WIDTH = 12
        self.TOP_K_OPS = 6
        
    def make_predictions(self, arc_problem: ArcProblem):
        
        
        
        def grids_equal(a: np.ndarray, b: np.ndarray):
            return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
            
        def most_freq(arr: np.ndarray):
            if arr.size == 0:
                return 0
            v, c = np.unique(arr, return_counts=True)
            return int(v[c.argmax()])
            
        def pixel_loss(a: np.ndarray, b: np.ndarray):
            if a.shape != b.shape:
                return 10**9
            return int(np.sum(a != b))
        def shape_match(a: np.ndarray, b: np.ndarray):
            return int(a.shape == b.shape)
        def cc4_info(g: np.ndarray):
            H, W = g.shape
            seen = np.zeros_like(g, dtype=bool)
            comps = []
            for r in range(H):
                for c in range(W):
                    if g[r, c] == 0 or seen[r, c]:
                        continue
                    col = int(g[r, c])
                    stack = [(r, c)]
                    seen[r, c] = True
                    coords = []
                    sr = sc = n = 0
                    minr = minc = 10**9
                    maxr = maxc = -10**9
                    while stack:
                        y, x = stack.pop()
                        coords.append((y, x))
                        sr += y; sc += x; n += 1
                        if y<minr: minr=y
                        if y>maxr: maxr=y
                        if x<minc: minc=x
                        if x>maxc: maxc=x
                        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                            ny, nx = y+dy, x+dx
                            if 0<=ny<H and 0<=nx<W and not seen[ny, nx] and g[ny, nx]==col:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                    cy, cx = sr/n, sc/n
                    comps.append({"coords": coords,
                                  "color": col,
                                  "size": n,
                                  "bbox": (minr, minc, maxr, maxc),
                                  "centroid": (cy, cx)})
            return comps
            
            
        def apply_seq(g: np.ndarray, seq: list):
            out = g
            for op in seq:
                out = op(out)      
            return out
        
        def sequence_fits_all(train_pairs: list, seq: list):
            tot = 0
            fails = 0
            for x, y in train_pairs:
                pred = apply_seq(x, seq)
                L = pixel_loss(pred, y)
                tot += L
                if L > 0:
                    fails += 1
            return tot, fails
        def dihedral_names():
            return ("I","R90","R180","R270","FH","FV","FD","FA")
        def op_dihedral(name: str):
            def f(g: np.ndarray):
                if name == "I":   return g.copy()
                if name == "R90": return np.rot90(g, 1)
                if name == "R180":return np.rot90(g, 2)
                if name == "R270":return np.rot90(g, 3)
                if name == "FH":  return np.flip(g, 1)
                if name == "FV":  return np.flip(g, 0)
                if name == "FD":  return np.transpose(g)
                if name == "FA":  return np.flip(np.transpose(g), 1)
                return g.copy()
            return f
            
            
        def compress_left(a: np.ndarray, bg: int=0):
            H, W = a.shape
            out = np.full_like(a, bg)
            for r in range(H):
                nz = a[r][a[r] != bg]
                if nz.size:
                    out[r, :nz.size] = nz
            return out
        def compress_up(a: np.ndarray, bg: int=0):
            H, W = a.shape
            out = np.full_like(a, bg)
            for c in range(W):
                col = a[:, c]
                nz = col[col != bg]
                if nz.size:
                    out[:nz.size, c] = nz
            return out
            
        def op_compress_left(bg: int=0): return lambda g: compress_left(g, bg)
        def op_compress_up(bg: int=0):   return lambda g: compress_up(g, bg)
        
        
        def translate(a: np.ndarray, dy: int, dx: int, bg: int=0):
            H, W = a.shape
            out = np.full_like(a, bg)
            y0 = max(0, dy); x0 = max(0, dx)
            y1 = min(H, H+dy); x1 = min(W, W+dx)
            sy0 = -min(0, dy); sx0 = -min(0, dx)
            if y0 < y1 and x0 < x1:
                out[y0:y1, x0:x1] = a[sy0:sy0+(y1-y0), sx0:sx0+(x1-x0)]
            return out
        def op_translate(dy: int, dx: int, bg: int=0): return lambda g: translate(g, dy, dx, bg)
        
        
        def object_repaint(a: np.ndarray, color: int):
            out = a.copy()
            comps = cc4_info(a)
            if not comps:
                return out
            comps.sort(key=lambda t: t["size"], reverse=True)
            for (r,c) in comps[0]["coords"]:
                out[r, c] = color
            return out
        def op_object_repaint(color: int): return lambda g: object_repaint(g, color)
        
        def cmap_from_xy(x: np.ndarray, y: np.ndarray):
            if x.shape != y.shape:
                return None
            cm = {}
            for xv, yv in zip(x.ravel(), y.ravel()):
                xv = int(xv); yv = int(yv)
                if xv not in cm:
                    cm[xv] = yv
                elif cm[xv] != yv:
                    return None
            return cm
        def apply_cmap(arr: np.ndarray, cmap: dict[int,int]):
            table = np.arange(10, dtype=arr.dtype)
            for k,v in cmap.items():
                if 0<=k<10 and 0<=v<10:
                    table[k] = v
            return table[arr]
            
        def repeat_upscale(a: np.ndarray, kh: int, kw: int):
            return np.kron(a, np.ones((kh, kw), dtype=a.dtype))
            
        def try_infer_repeat_xy(train_pairs):
            kh_set, kw_set = set(), set()
            for x, y in train_pairs:
                Hx, Wx = x.shape
                Hy, Wy = y.shape
                if Hy % Hx != 0 or Wy % Wx != 0:
                    return None
                kh, kw = Hy // Hx, Wy // Wx
                if not grids_equal(repeat_upscale(x, kh, kw), y):
                    return None
                kh_set.add(kh); kw_set.add(kw)
            if len(kh_set) == 1 and len(kw_set) == 1:
                return (next(iter(kh_set)), next(iter(kw_set)))
            return None
        def block_downscale_mode_zero_exclusive(a: np.ndarray, h: int, w: int):
            H, W = a.shape
            assert H % h == 0 and W % w == 0
            bh, bw = H//h, W//w
            out = np.zeros((h,w), dtype=a.dtype)
            for i in range(h):
                for j in range(w):
                    tile = a[i*bh:(i+1)*bh, j*bw:(j+1)*bw].ravel()
                    tile = tile[tile != 0]
                    if tile.size == 0:
                        out[i,j] = 0
                    else:
                        v, c = np.unique(tile, return_counts=True)
                        out[i,j] = int(v[c.argmax()])
            return out
        def block_downscale_mode_zero_inclusive(a: np.ndarray, h: int, w: int):
            H, W = a.shape
            assert H % h == 0 and W % w == 0
            bh, bw = H // h, W // w
            out = np.zeros((h, w), dtype=a.dtype)
            for i in range(h):
                for j in range(w):
                    tile = a[i*bh:(i+1)*bh, j*bw:(j+1)*bw].ravel()
                    v, c = np.unique(tile, return_counts=True)
                    out[i, j] = int(v[c.argmax()])
            return out
        def try_infer_downscale(train_pairs, sizes=((4,4), (3,3), (2,2))):
            for (hh, ww) in sizes:
                ok = True
                for x, y in train_pairs:
                    Hx, Wx = x.shape
                    if Hx % hh != 0 or Wx % ww != 0:
                        ok = False; break
                    if not grids_equal(block_downscale_mode_zero_exclusive(x, hh, ww), y):
                        ok = False; break
                if ok:
                    return lambda g, hh=hh, ww=ww: block_downscale_mode_zero_exclusive(g, hh, ww)
                ok = True
                for x, y in train_pairs:
                    Hx, Wx = x.shape
                    if Hx % hh != 0 or Wx % ww != 0:
                        ok = False; break
                    if not grids_equal(block_downscale_mode_zero_inclusive(x, hh, ww), y):
                        ok = False; break
                if ok:
                    return lambda g, hh=hh, ww=ww: block_downscale_mode_zero_inclusive(g, hh, ww)
            return None
        def fill_lines_by_anchors(a: np.ndarray):
            g = a.copy()
            H, W = g.shape
            out = g.copy()
            for r in range(H):
                row = g[r]
                nz = np.where(row != 0)[0]
                if nz.size >= 2:
                    left, right = nz[0], nz[-1]
                    if row[left] == row[right]:
                        out[r, :] = np.where(out[r,:]==0, row[left], out[r, :])
            for c in range(W):
                col = g[:, c]
                nz = np.where(col != 0)[0]
                if nz.size >= 2:
                    top, bot = nz[0], nz[-1]
                    if col[top] == col[bot]:
                        out[:, c] = np.where(out[:,c]==0, col[top], out[:,c])
            return out
        
        def _center_component(g: np.ndarray):
            H, W = g.shape
            cy, cx = (H-1)/2.0, (W-1)/2.0
            comps = cc4_info(g)
            if not comps:
                return None
            candidates = []
            for c in comps:
                r0, c0, r1, c1 = c["bbox"]
                if r0 <= cy <= r1 and c0 <= cx <= c1:
                    candidates.append(c)
            if candidates:
                candidates.sort(key=lambda c: (c["centroid"][0]-cy)**2 + (c["centroid"][1]-cx)**2)
                return candidates[0]
            comps.sort(key=lambda c: (c["centroid"][0]-cy)**2 + (c["centroid"][1]-cx)**2)
            return comps[0]
        
        def spine_fill_through_centroid(a: np.ndarray) -> np.ndarray:
            g = a.copy()
            comp = _center_component(g)
            if comp is None:
                return g
            colors = [g[r, c] for (r, c) in comp["coords"]]
            col = int(np.argmax(np.bincount(np.array(colors, dtype=int), minlength=10)))
            cy, cx = comp["centroid"]
            cxr = int(round(cx))
            out = g.copy()
            out[:, cxr] = np.where(out[:, cxr]==0, col, out[:, cxr])
            return out

        def extend_diagonals_from_seeds(a: np.ndarray):
            g = a.copy()
            H, W = g.shape
            out = g.copy()
            dirs = [(-1,1), (1,1), (-1,-1), (1,-1)]
            for comp in cc4_info(g):
                col = comp["color"]
                S = set(comp["coords"])
                for (y, x) in comp["coords"]:
                    for dy, dx in dirs:
                        py, px = y - dy, x - dx
                        if (py, px) not in S:
                            ny, nx = y, x
                            while 0 <= ny < H and 0 <= nx < W:
                                if out[ny,nx] == 0:
                                    out[ny,nx] = col
                                ny += dy; nx += dx
            return out
        def staircase_from_edge_seed(a: np.ndarray):
            g = a.copy()
            H, W = g.shape
            out = np.zeros_like(g)
            def run_len(vec):
                idx = np.where(vec != 0)[0]
                if idx.size == 0:
                    return 0, 0
                start, end = idx[0], idx[-1]
                col = int(vec[idx[0]])
                if np.all(vec[start:end+1] == col):
                    return end-start+1, col
                return 0, 0
            Lr, cr = run_len(g[0, :])
            Lc, cc = run_len(g[:, 0])
            if Lr >= 2:
                for i in range(min(H, Lr)):
                    out[i, :Lr-i] = cr
                return out
            if Lc >= 2:
                for j in range(min(W, Lc)):
                    out[:Lc-j, j] = cc
                return out
            return g.copy()
            
        def op_score(op, train_pairs: list[tuple[np.ndarray, np.ndarray]]):
            exact = 0
            total_loss = 0
            shape_matches = 0
            def pal(a): return set(np.unique(a))
            color_penalty = 0
            for x, y in train_pairs:
                try:
                    py = op(x)
                except Exception:
                    return (10**9, 10**9, -10**9, 10**9)
                if grids_equal(py, y):
                    exact += 1
                total_loss += pixel_loss(py,y)
                shape_matches += shape_match(py, y)
                color_penalty += len(pal(py) ^ pal(y))
            return (-exact, total_loss, -shape_matches, color_penalty)
        train_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for s in arc_problem.training_set():
            x = s.get_input_data().data().copy()
            y = s.get_output_data().data().copy()
            train_pairs.append((x, y))
        test_in = arc_problem.test_set().get_input_data().data().copy()
        
        gated_ops: list = []
        

        for name in dihedral_names():
            op = op_dihedral(name)
            try:
                if all(grids_equal(op(x), y) for x, y in train_pairs):
                    gated_ops.append(op)
            except Exception:
                pass
                
        try:
            if all(grids_equal(compress_left(x, 0), y) for x, y in train_pairs):
                gated_ops.append(op_compress_left(0))
        except Exception:
            pass
        try:
            if all(grids_equal(compress_up(x, 0), y) for x, y in train_pairs):
                gated_ops.append(op_compress_up(0))
        except Exception:
            pass
        nonzero_pool = [y[y != 0] for _x, y in train_pairs if np.any(y != 0)]
        dom_out = most_freq(np.concatenate(nonzero_pool)) if nonzero_pool else 0
        try:
            if all(grids_equal(object_repaint(x, dom_out), y) for x, y in train_pairs):
                gated_ops.append(op_object_repaint(dom_out))
        except Exception:
            pass
        gmap = {}
        ok_map = True
        for x, y in train_pairs:
            cm = cmap_from_xy(x, y)
            if cm is None:
                ok_map = False
                break
            for k,v in cm.items():
                if k in gmap and gmap[k] != v:
                    ok_map = False
                    break
                gmap[k] = v
            if not ok_map:
                break
        if ok_map and gmap:
            def cmap_op(g, cm=gmap): return apply_cmap(g, cm)
            try:
                if all(grids_equal(cmap_op(x), y) for x,y in train_pairs):
                    gated_ops.append(cmap_op)
            except Exception:
                pass
        
        rpt = try_infer_repeat_xy(train_pairs)
        if rpt is not None:
            kh, kw = rpt
            def rpt_op(g, kh=kh, kw=kw): return repeat_upscale(g, kh, kw)
            gated_ops.append(rpt_op)
        dn = try_infer_downscale(train_pairs)
        if dn is not None:
            gated_ops.append(dn)
            
        try:
            if all(grids_equal(fill_lines_by_anchors(x), y) for x, y in train_pairs):
                gated_ops.append(lambda g: fill_lines_by_anchors(g))
        except Exception:
            pass
        try:
            if all(grids_equal(spine_fill_through_centroid(x), y) for x,y in train_pairs):
                gated_ops.append(lambda g: spine_fill_through_centroid(g))
        except Exception:
            pass
            
        try:
            if all(grids_equal(extend_diagonals_from_seeds(x), y) for x,y in train_pairs):
                gated_ops.append(lambda g: extend_diagonals_from_seeds(g))
        except Exception:
            pass
            
        try:
            if all(grids_equal(staircase_from_edge_seed(x), y) for x,y in train_pairs):
                gated_ops.append(lambda g: staircase_from_edge_seed(g))
        except Exception:
            pass
            
        preds = []
        seen = set()
        def add_pred(p: np.ndarray):
            for q in preds:
                if grids_equal(p, q):
                    return
            if len(preds) < 3:
                preds.append(p)
        if gated_ops:
            for op in gated_ops:
                try:
                    p = op(test_in)
                    key = (p.shape, tuple(p.ravel().tolist()[:64]))
                    if key in seen:
                        continue
                    seen.add(key)
                    add_pred(p)
                    if len(preds) >= 3:
                        break
                except Exception:
                    continue
            if preds:
                return preds[:3]
                
                
        candidate_ops = [op_dihedral(n) for n in dihedral_names()]
        candidate_ops += [op_compress_left(0), op_compress_up(0)]
        candidate_ops += [lambda g: fill_lines_by_anchors(g),
                          lambda g: spine_fill_through_centroid(g),
                          lambda g: extend_diagonals_from_seeds(g),
                          lambda g: staircase_from_edge_seed(g)]
      
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                candidate_ops.append(op_translate(dy, dx, 0))
                
        if ok_map and gmap:
            candidate_ops.append(lambda g, cm=gmap: apply_cmap(g, cm))
        if nonzero_pool:
            candidate_ops.append(op_object_repaint(dom_out))
            
        
        scored = []
        for op in candidate_ops:
            try:
                s = op_score(op, train_pairs)
            except Exception:
                s = (10**9, 10**9, -10**9, 10**9)
            scored.append((s, op))
            
        scored.sort(key=lambda t: t[0])
        top_ops = [op for (_s, op) in scored[:self.TOP_K_OPS]]
        
        for op in top_ops:
            try:
                p = op(test_in)
                add_pred(p)
                if len(preds) >= 3:
                    break
            except Exception:
                continue
                
        if preds:
            return preds[:3]
            
        core_ops = [op_dihedral(n) for n in dihedral_names()]
        core_ops += [op_compress_left(0), op_compress_up(0)]
        for dy in (-1, 1):
            for dx in (-1, 1):
                core_ops.append(op_translate(dy, dx, 0))
        BeamItem = tuple[int, list]
        beam: list[BeamItem] = [(0, [])]
        best_exact: list[list] = []
        best_approx: list[tuple[int, list]] = []
        
        L0, _ = sequence_fits_all(train_pairs, [])
        if L0 == 0:
            best_exact.append([])
            
        for _depth in range(1, self.MAX_DEPTH + 1):
            new_beam: list[BeamItem] = []
            for _cur_loss, seq in beam:
                for op in core_ops:
                    seq2 = seq + [op]
                    L, _fails = sequence_fits_all(train_pairs, seq2)
                    new_beam.append((L, seq2))
            new_beam.sort(key=lambda t: t[0])
            beam = new_beam[:self.BEAM_WIDTH]
            for L, seq in beam:
                if L == 0:
                    best_exact.append(seq)
                else:
                    best_approx.append((L, seq))

        for seq in best_exact[:3]:
            add_pred(apply_seq(test_in, seq))
            if len(preds) >= 3:
                break
        if len(preds) < 3 and best_approx:
            best_approx.sort(key=lambda t: t[0])
            for L, seq in best_approx[:3]:
                add_pred(apply_seq(test_in, seq))
                if len(preds) >= 3:
                    break
               
        return preds[:3] if preds else[test_in.copy()]
