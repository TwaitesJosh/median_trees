"""
Median trees: random trees cut at data quantiles, a sparse linear head on their leaves.

Implementation module. The public API is re-exported from ``median_trees``:
MedianTreeClassifier, MedianTreeEmbedder, TunedMedianTreeClassifier, LeafSGD.

Numba kernels (tree building, leaf lookup, lock-free parallel SGD) are compiled on first use
and cached next to this file.
"""

import warnings
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

_CHUNK = 4096
_N_HOGWILD_BLOCKS = 32  # parallel row blocks per SGD epoch

try:
    import numba

    _NJIT_PAR = dict(parallel=True, cache=True, boundscheck=False)
    _NJIT_PAR_FAST = dict(parallel=True, cache=True, fastmath=True, boundscheck=False)
    _NJIT_SEQ = dict(cache=True, boundscheck=False)
    _NJIT_SEQ_FAST = dict(cache=True, fastmath=True, boundscheck=False)

    # -------------------------------------------------------------------------
    # NaN-Aware Median Selection & Tree Evaluation
    # -------------------------------------------------------------------------
    @numba.njit(**_NJIT_SEQ)
    def _kth_smallest_inplace(v, k):
        lo, hi = 0, len(v) - 1
        while lo < hi:
            pivot = v[(lo + hi) >> 1]
            i, j = lo, hi
            while i <= j:
                while v[i] < pivot:
                    i += 1
                while v[j] > pivot:
                    j -= 1
                if i <= j:
                    v[i], v[j] = v[j], v[i]
                    i += 1
                    j -= 1
            if k <= j:
                hi = j
            elif k >= i:
                lo = i
            else:
                return v[k]
        return v[k]

    @numba.njit(**_NJIT_SEQ)
    def _select_threshold_nan(v, cnt, q):
        """Filter out NaNs in v[:cnt], compute quantile q of valid values, and return
        (ok, threshold, n_valid)."""
        n_val = 0
        for j in range(cnt):
            x = v[j]
            if not np.isnan(x):
                v[n_val] = x
                n_val += 1
        if n_val <= 1:
            return False, np.float32(0.0), n_val
        lo, hi = v[0], v[0]
        for j in range(1, n_val):
            x = v[j]
            if x < lo:
                lo = x
            elif x > hi:
                hi = x
        if lo == hi:
            return False, lo, n_val
        thr = _kth_smallest_inplace(v[:n_val], int(q * (n_val - 1)))
        if thr == hi:
            thr = lo
            for j in range(n_val):
                x = v[j]
                if x < hi and x > thr:
                    thr = x
        return True, thr, n_val

    @numba.njit(**_NJIT_PAR)
    def _apply_full_nb(X, feature1, feature2, threshold, nan_right, depth):
        """Routes rows through trees, sending NaNs to `nan_right[t, node]` (0=left, 1=right)."""
        n, T = X.shape[0], feature1.shape[0]
        out = np.empty((n, T), dtype=np.int32)
        offset = np.int32((1 << depth) - 1)
        for i in numba.prange(n):
            for t in range(T):
                node = np.int32(0)
                for _ in range(depth):
                    f1 = feature1[t, node]
                    f2 = feature2[t, node]
                    val = X[i, f1] if f2 < 0 else (X[i, f1] - X[i, f2])
                    if np.isnan(val):
                        go_r = np.int32(nan_right[t, node])
                    else:
                        go_r = np.int32(val > threshold[t, node])
                    node = (node << np.int32(1)) + np.int32(1) + go_r
                out[i, t] = node - offset
        return out

    @numba.njit(**_NJIT_PAR)
    def _apply_oblivious_nb(X, feature, threshold):
        n, T, D = X.shape[0], feature.shape[0], feature.shape[1]
        out = np.empty((n, T), dtype=np.int32)
        for i in numba.prange(n):
            for t in range(T):
                leaf = np.int32(0)
                for j in range(D):
                    val = X[i, feature[t, j]]
                    if not np.isnan(val) and val > threshold[t, j]:
                        leaf |= np.int32(1) << np.int32(j)
                out[i, t] = leaf
        return out

    # -------------------------------------------------------------------------
    # Parallel Inference Lookup Kernels
    # -------------------------------------------------------------------------
    @numba.njit(**_NJIT_PAR_FAST)
    def _lookup_predict_bin_nb(leaves, depth, W1, b1):
        n, T = leaves.shape[0], leaves.shape[1]
        out = np.empty(n, dtype=np.float32)
        sh = np.int32(depth)
        for i in numba.prange(n):
            acc = b1
            for t in range(T):
                acc += W1[(np.int32(t) << sh) | leaves[i, t]]
            out[i] = acc
        return out

    @numba.njit(**_NJIT_PAR_FAST)
    def _lookup_predict_nb(leaves, depth, W, b):
        n, T, C = leaves.shape[0], leaves.shape[1], W.shape[1]
        out = np.empty((n, C), dtype=np.float32)
        sh = np.int32(depth)
        for i in numba.prange(n):
            for c in range(C):
                out[i, c] = b[c]
            for t in range(T):
                row = (np.int32(t) << sh) | leaves[i, t]
                for c in range(C):
                    out[i, c] += W[row, c]
        return out

    # -------------------------------------------------------------------------
    # Lock-Free Parallel Hogwild! SGD Kernels (Multi-Core Accelerated)
    # -------------------------------------------------------------------------
    # Each parallel epoch splits the shuffled rows into blocks and runs plain sequential SGD
    # on every block in its own thread, all writing to the shared weights (Hogwild!).
    # The per-block work lives in separate non-parallel helpers: Numba's parfor pass
    # mis-analyses in-place updates such as `state ^= ...` inside a prange body and can
    # recurse forever (RecursionError at compile time). Calling a helper avoids that.

    @numba.njit(**_NJIT_SEQ_FAST)
    def _upd1(W1, G2, row, g, lr, l2, l2w, adagrad):
        """Update one binary leaf weight. l2w[row] scales the L2 term per weight (ones = plain
        decay on every visit; 1/count = ridge-style, the same total decay for every leaf
        per epoch). With adagrad, G2 accumulates squared gradients per weight."""
        gr = g + l2 * l2w[row] * W1[row]
        if adagrad:
            G2[row] += gr * gr
            W1[row] -= lr * gr / np.sqrt(G2[row])
        else:
            W1[row] -= lr * gr

    @numba.njit(**_NJIT_SEQ_FAST)
    def _updC(W, G2, row, g, lr, l2, l2w, adagrad):
        """Multiclass version of _upd1 (one weight per class)."""
        C = W.shape[1]
        w = l2 * l2w[row]
        for c in range(C):
            gr = g[c] + w * W[row, c]
            if adagrad:
                G2[row, c] += gr * gr
                W[row, c] -= lr * gr / np.sqrt(G2[row, c])
            else:
                W[row, c] -= lr * gr

    @numba.njit(**_NJIT_SEQ_FAST)
    def _rng_init(seed, blk):
        state = np.uint32((seed + 1) * 1664525 + blk * 1013904223 + 1)
        if state == 0:
            state = np.uint32(2463534242)
        return state

    @numba.njit(**_NJIT_SEQ_FAST)
    def _blk_bin(leaves, y, order, start, end, sh, W1, b_init, O1,
                 lr, lr_scaled, l2, softmax, drop_p, thresh, scale, seed, blk, l2w, G2, adagrad):
        T = leaves.shape[1]
        b_loc = b_init
        db_acc = np.float32(0.0)
        state = _rng_init(seed, blk)
        active = np.empty(T, dtype=np.int32)
        for idx_pos in range(start, end):
            i = order[idx_pos]
            n_act = 0
            if drop_p <= 0.0:
                s = b_loc + O1[i]
                for t in range(T):
                    s += W1[(np.int32(t) << sh) | leaves[i, t]]
            else:
                s = np.float32(0.0)
                for t in range(T):
                    state ^= state << np.uint32(13)
                    state ^= state >> np.uint32(17)
                    state ^= state << np.uint32(5)
                    if state >= thresh:
                        row = (np.int32(t) << sh) | leaves[i, t]
                        active[n_act] = row
                        n_act += 1
                        s += W1[row]
                s = b_loc + O1[i] + s * scale
            if softmax:
                p = np.float32(1.0) / (np.float32(1.0) + np.exp(-s))
                g = p - np.float32(y[i])
            else:
                target = np.float32(1.0) if y[i] == 1 else np.float32(-1.0)
                g = s - target
            if drop_p <= 0.0:
                for t in range(T):
                    row = (np.int32(t) << sh) | leaves[i, t]
                    _upd1(W1, G2, row, g, lr, l2, l2w, adagrad)
            else:
                for k in range(n_act):
                    _upd1(W1, G2, active[k], g, lr_scaled, l2, l2w, adagrad)
            step_b = lr * g
            b_loc -= step_b
            db_acc -= step_b
        return db_acc

    @numba.njit(**_NJIT_PAR_FAST)
    def _sgd_epoch_bin_nb(leaves, y, order, depth, W1, b1, O1, lr, l2, softmax, drop_p, seed,
                          l2w, G2, adagrad):
        """Lock-free parallel Hogwild! 1D binary SGD across CPU cores."""
        n = len(order)
        sh = np.int32(depth)
        # Binary weights are 1 float per leaf, so 16 leaves of a tree share a cache line and
        # parallel threads thrash it (false sharing): measured 2x SLOWER than one thread.
        # Binary epochs are cheap anyway, so they run as a single block (= sequential SGD).
        n_blocks = 1
        block_size = n
        db_blocks = np.zeros(n_blocks, dtype=np.float32)
        b_init = b1[0]
        scale = np.float32(1.0 / (1.0 - drop_p)) if drop_p > 0.0 else np.float32(1.0)
        lr_scaled = lr * scale
        thresh = np.uint32(int(drop_p * 4294967295.0)) if drop_p > 0.0 else np.uint32(0)
        for blk in numba.prange(n_blocks):
            db_blocks[blk] = _blk_bin(leaves, y, order, blk * block_size,
                                      min(n, (blk + 1) * block_size), sh, W1, b_init, O1,
                                      lr, lr_scaled, l2, softmax, drop_p, thresh, scale, seed, blk,
                                      l2w, G2, adagrad)
        b1[0] = b_init + db_blocks.sum() / np.float32(n_blocks)

    @numba.njit(**_NJIT_SEQ_FAST)
    def _blk_bin_raw(leaves, X, y, order, start, end, sh, W1, V1, b_init, O1,
                     lr, lr_raw, lr_scaled, l2, softmax, drop_p, thresh, scale, seed, blk,
                     l2_raw, l2w, G2, adagrad):
        T, d = leaves.shape[1], X.shape[1]
        b_loc = b_init
        db_acc = np.float32(0.0)
        state = _rng_init(seed, blk)
        active = np.empty(T, dtype=np.int32)
        for idx_pos in range(start, end):
            i = order[idx_pos]
            n_act = 0
            s = np.float32(0.0)
            if drop_p <= 0.0:
                for t in range(T):
                    s += W1[(np.int32(t) << sh) | leaves[i, t]]
            else:
                for t in range(T):
                    state ^= state << np.uint32(13)
                    state ^= state >> np.uint32(17)
                    state ^= state << np.uint32(5)
                    if state >= thresh:
                        row = (np.int32(t) << sh) | leaves[i, t]
                        active[n_act] = row
                        n_act += 1
                        s += W1[row]
                s *= scale
            s += b_loc + O1[i]
            for j in range(d):
                s += X[i, j] * V1[j]
            if softmax:
                p = np.float32(1.0) / (np.float32(1.0) + np.exp(-s))
                g = p - np.float32(y[i])
            else:
                target = np.float32(1.0) if y[i] == 1 else np.float32(-1.0)
                g = s - target
            if drop_p <= 0.0:
                for t in range(T):
                    row = (np.int32(t) << sh) | leaves[i, t]
                    _upd1(W1, G2, row, g, lr, l2, l2w, adagrad)
            else:
                for k in range(n_act):
                    _upd1(W1, G2, active[k], g, lr_scaled, l2, l2w, adagrad)
            for j in range(d):
                V1[j] -= lr_raw * (g * X[i, j] + l2_raw * V1[j])
            step_b = lr * g
            b_loc -= step_b
            db_acc -= step_b
        return db_acc

    @numba.njit(**_NJIT_PAR_FAST)
    def _sgd_epoch_bin_raw_nb(leaves, X, y, order, depth, W1, V1, b1, O1,
                              lr, lr_raw, l2, softmax, drop_p, seed, l2_raw, l2w, G2, adagrad):
        n = len(order)
        sh = np.int32(depth)
        # Binary weights are 1 float per leaf, so 16 leaves of a tree share a cache line and
        # parallel threads thrash it (false sharing): measured 2x SLOWER than one thread.
        # Binary epochs are cheap anyway, so they run as a single block (= sequential SGD).
        n_blocks = 1
        block_size = n
        db_blocks = np.zeros(n_blocks, dtype=np.float32)
        b_init = b1[0]
        scale = np.float32(1.0 / (1.0 - drop_p)) if drop_p > 0.0 else np.float32(1.0)
        lr_scaled = lr * scale
        thresh = np.uint32(int(drop_p * 4294967295.0)) if drop_p > 0.0 else np.uint32(0)
        for blk in numba.prange(n_blocks):
            db_blocks[blk] = _blk_bin_raw(leaves, X, y, order, blk * block_size,
                                          min(n, (blk + 1) * block_size), sh, W1, V1, b_init, O1,
                                          lr, lr_raw, lr_scaled, l2, softmax, drop_p, thresh,
                                          scale, seed, blk, l2_raw, l2w, G2, adagrad)
        b1[0] = b_init + db_blocks.sum() / np.float32(n_blocks)

    @numba.njit(**_NJIT_SEQ_FAST)
    def _softmax_grad(s, g, yi, softmax):
        C = len(s)
        if softmax:
            m = s[0]
            for c in range(1, C):
                if s[c] > m:
                    m = s[c]
            z = np.float32(0.0)
            for c in range(C):
                g[c] = np.exp(s[c] - m)
                z += g[c]
            inv_z = np.float32(1.0) / z
            for c in range(C):
                g[c] *= inv_z
            g[yi] -= np.float32(1.0)
        else:
            for c in range(C):
                g[c] = s[c] + np.float32(1.0)
            g[yi] = s[yi] - np.float32(1.0)

    @numba.njit(**_NJIT_SEQ_FAST)
    def _blk_mc(leaves, X, y, order, start, end, sh, W, V, b_init, O,
                lr, lr_raw, lr_scaled, l2, softmax, drop_p, thresh, scale, seed, blk,
                db_row, use_raw, l2_raw, l2w, G2, adagrad):
        """One block of multiclass SGD; X/V are only used when use_raw is True."""
        T, C = leaves.shape[1], W.shape[1]
        d = X.shape[1]
        s = np.empty(C, dtype=np.float32)
        g = np.empty(C, dtype=np.float32)
        b_loc = b_init.copy()
        state = _rng_init(seed, blk)
        active = np.empty(T, dtype=np.int32)
        for idx_pos in range(start, end):
            i = order[idx_pos]
            n_act = 0
            for c in range(C):
                s[c] = np.float32(0.0)
            if drop_p <= 0.0:
                for t in range(T):
                    row = (np.int32(t) << sh) | leaves[i, t]
                    for c in range(C):
                        s[c] += W[row, c]
            else:
                for t in range(T):
                    state ^= state << np.uint32(13)
                    state ^= state >> np.uint32(17)
                    state ^= state << np.uint32(5)
                    if state >= thresh:
                        row = (np.int32(t) << sh) | leaves[i, t]
                        active[n_act] = row
                        n_act += 1
                        for c in range(C):
                            s[c] += W[row, c]
                for c in range(C):
                    s[c] *= scale
            for c in range(C):
                s[c] += b_loc[c] + O[i, c]
            if use_raw:
                for j in range(d):
                    xj = X[i, j]
                    for c in range(C):
                        s[c] += xj * V[j, c]
            _softmax_grad(s, g, y[i], softmax)
            if drop_p <= 0.0:
                for t in range(T):
                    row = (np.int32(t) << sh) | leaves[i, t]
                    _updC(W, G2, row, g, lr, l2, l2w, adagrad)
            else:
                for k in range(n_act):
                    _updC(W, G2, active[k], g, lr_scaled, l2, l2w, adagrad)
            if use_raw:
                for j in range(d):
                    xj = X[i, j]
                    for c in range(C):
                        V[j, c] -= lr_raw * (g[c] * xj + l2_raw * V[j, c])
            for c in range(C):
                step_b = lr * g[c]
                b_loc[c] -= step_b
                db_row[c] -= step_b

    @numba.njit(**_NJIT_PAR_FAST)
    def _sgd_epoch_mc_core(leaves, X, y, order, depth, W, V, b, O, lr, lr_raw, l2,
                           softmax, drop_p, seed, use_raw, l2_raw, l2w, G2, adagrad):
        n = len(order)
        C = W.shape[1]
        sh = np.int32(depth)
        n_blocks = min(_N_HOGWILD_BLOCKS, max(1, n // 128))
        block_size = (n + n_blocks - 1) // n_blocks
        db_blocks = np.zeros((n_blocks, C), dtype=np.float32)
        b_init = b.copy()
        scale = np.float32(1.0 / (1.0 - drop_p)) if drop_p > 0.0 else np.float32(1.0)
        lr_scaled = lr * scale
        thresh = np.uint32(int(drop_p * 4294967295.0)) if drop_p > 0.0 else np.uint32(0)
        for blk in numba.prange(n_blocks):
            _blk_mc(leaves, X, y, order, blk * block_size, min(n, (blk + 1) * block_size), sh,
                    W, V, b_init, O, lr, lr_raw, lr_scaled, l2, softmax, drop_p, thresh, scale,
                    seed, blk, db_blocks[blk], use_raw, l2_raw, l2w, G2, adagrad)
        for c in range(C):
            acc_b = np.float32(0.0)
            for k in range(n_blocks):
                acc_b += db_blocks[k, c]
            b[c] = b_init[c] + acc_b / np.float32(n_blocks)

    def _sgd_epoch_nb(leaves, y, order, depth, W, b, O, lr, l2, softmax, drop_p, seed,
                      l2w, G2, adagrad):
        """Lock-free parallel Hogwild! multiclass SGD across CPU cores (leaves only)."""
        dummy_X = np.zeros((1, 1), dtype=np.float32)
        dummy_V = np.zeros((1, W.shape[1]), dtype=np.float32)
        _sgd_epoch_mc_core(leaves, dummy_X, y, order, depth, W, dummy_V, b, O, lr,
                           np.float32(0.0), l2, softmax, drop_p, seed, False,
                           np.float32(0.0), l2w, G2, adagrad)

    def _sgd_epoch_raw_nb(leaves, X, y, order, depth, W, V, b, O, lr, lr_raw, l2, softmax,
                          drop_p, seed, l2_raw, l2w, G2, adagrad):
        """Lock-free parallel Hogwild! multiclass SGD with a dense raw-feature term."""
        _sgd_epoch_mc_core(leaves, X, y, order, depth, W, V, b, O, lr, lr_raw, l2,
                           softmax, drop_p, seed, True, l2_raw, l2w, G2, adagrad)

    @numba.njit(**_NJIT_SEQ)
    def _leaf_stats_nb(leaves, sh, G, n_rows):
        """Per weight row (tree, leaf): number of training rows and summed targets G."""
        n, T = leaves.shape
        K = G.shape[1]
        cnt = np.zeros(n_rows, dtype=np.float32)
        S = np.zeros((n_rows, K), dtype=np.float32)
        for i in range(n):
            for t in range(T):
                row = (np.int32(t) << sh) | leaves[i, t]
                cnt[row] += 1.0
                for k in range(K):
                    S[row, k] += G[i, k]
        return cnt, S

    # -------------------------------------------------------------------------
    # Tree Fitting Kernels with Learned Missing-Value (NaN) Routing
    # -------------------------------------------------------------------------
    @numba.njit(**_NJIT_PAR)
    def _fit_full_nb(XT, boot, cand1, cand2, qs, depth, feature1, feature2, threshold, nan_right):
        """Unsupervised median splits. Ignores NaNs when finding the median of observed values,
        and routes NaNs to whichever side has fewer observed samples to keep leaves balanced."""
        T, m = boot.shape
        n_internal = (1 << depth) - 1
        n_try = cand1.shape[2]
        for t in numba.prange(T):
            idx = boot[t].copy()
            lo = np.zeros(2 * n_internal + 1, dtype=np.int32)
            hi = np.zeros(2 * n_internal + 1, dtype=np.int32)
            hi[0] = m
            vals = np.empty(m, dtype=XT.dtype)
            raw_vals = np.empty(m, dtype=XT.dtype)
            for node in range(n_internal):
                a, b = lo[node], hi[node]
                split = a
                if b - a > 1:
                    found = False
                    cnt = b - a
                    for k in range(n_try):
                        f1 = cand1[t, node, k]
                        f2 = cand2[t, node, k]
                        if f2 < 0:
                            for j in range(cnt):
                                v = XT[f1, idx[a + j]]
                                vals[j] = v
                                raw_vals[j] = v
                        else:
                            for j in range(cnt):
                                r = idx[a + j]
                                v = XT[f1, r] - XT[f2, r]
                                vals[j] = v
                                raw_vals[j] = v
                        ok, thr, n_val = _select_threshold_nan(vals, cnt, qs[t, node])
                        if not ok:
                            continue
                        # Count observed left vs right to route NaNs to the smaller side
                        n_obs_left = 0
                        for j in range(cnt):
                            rv = raw_vals[j]
                            if not np.isnan(rv) and rv <= thr:
                                n_obs_left += 1
                        n_obs_right = n_val - n_obs_left
                        nr_flag = np.int8(1 if n_obs_right < n_obs_left else 0)

                        feature1[t, node] = f1
                        feature2[t, node] = f2
                        threshold[t, node] = thr
                        nan_right[t, node] = nr_flag
                        p = a
                        for j in range(cnt):
                            rv = raw_vals[j]
                            go_left = (nr_flag == 0) if np.isnan(rv) else (rv <= thr)
                            if go_left:
                                idx[p], idx[a + j] = idx[a + j], idx[p]
                                p += 1
                        split = p
                        found = True
                        break
                    if not found:
                        split = b
                lo[2 * node + 1], hi[2 * node + 1] = a, split
                lo[2 * node + 2], hi[2 * node + 2] = split, b

    @numba.njit(**_NJIT_PAR)
    def _fit_sup_nb(XT, G, boot, cand1, cand2, qs, depth, score_rows, base_seed,
                    feature1, feature2, threshold, nan_right):
        """Supervised median splits with XGBoost/LightGBM-style missing value routing:
        evaluates routing NaNs left vs right on the target residuals G and picks the
        direction (and candidate feature) with maximum separation."""
        T, m = boot.shape
        C = G.shape[1]
        n_internal = (1 << depth) - 1
        n_try = cand1.shape[2]
        for t in numba.prange(T):
            idx = boot[t].copy()
            lo = np.zeros(2 * n_internal + 1, dtype=np.int32)
            hi = np.zeros(2 * n_internal + 1, dtype=np.int32)
            hi[0] = m
            vals = np.empty(m, dtype=XT.dtype)
            raw_vals = np.empty(m, dtype=XT.dtype)
            tot = np.empty(C, dtype=np.float32)
            left = np.empty(C, dtype=np.float32)
            nan_g = np.empty(C, dtype=np.float32)
            rng_state = np.uint32((base_seed + 1) * 1664525 + t * 1013904223 + 1)
            if rng_state == 0:
                rng_state = np.uint32(2463534242)

            for node in range(n_internal):
                a, b = lo[node], hi[node]
                split = a
                if b - a > 1:
                    cnt = b - a
                    sub = min(cnt, score_rows)
                    for c in range(C):
                        tot[c] = 0.0
                    for j in range(a, a + sub):
                        row_idx = idx[j]
                        for c in range(C):
                            tot[c] += G[row_idx, c]
                    best, best_f1, best_f2, best_nr = -1.0, -1, -1, np.int8(0)
                    for k in range(n_try):
                        f1 = cand1[t, node, k]
                        f2 = cand2[t, node, k]
                        if f2 < 0:
                            for j in range(sub):
                                v = XT[f1, idx[a + j]]
                                vals[j] = v
                                raw_vals[j] = v
                        else:
                            for j in range(sub):
                                r = idx[a + j]
                                v = XT[f1, r] - XT[f2, r]
                                vals[j] = v
                                raw_vals[j] = v
                        ok, thr, _ = _select_threshold_nan(vals, sub, qs[t, node])
                        if not ok:
                            continue
                        for c in range(C):
                            left[c] = 0.0
                            nan_g[c] = 0.0
                        n_left = 0
                        n_nan = 0
                        for j in range(sub):
                            rv = raw_vals[j]
                            row_idx = idx[a + j]
                            if np.isnan(rv):
                                n_nan += 1
                                for c in range(C):
                                    nan_g[c] += G[row_idx, c]
                            elif rv <= thr:
                                n_left += 1
                                for c in range(C):
                                    left[c] += G[row_idx, c]

                        # Try routing NaNs right (nr=1) and routing NaNs left (nr=0)
                        for nr_try in range(2):
                            nl = n_left if nr_try == 1 else (n_left + n_nan)
                            nr = sub - nl
                            if nl == 0 or nr == 0:
                                continue
                            inv_l = 1.0 / nl
                            inv_r = 1.0 / nr
                            score = 0.0
                            for c in range(C):
                                lc = left[c] if nr_try == 1 else (left[c] + nan_g[c])
                                rc = tot[c] - lc
                                score += lc * lc * inv_l + rc * rc * inv_r
                            if score > best:
                                best, best_f1, best_f2, best_nr = score, f1, f2, np.int8(nr_try)

                    ok_full = False
                    best_thr = np.float32(1e30)
                    if best_f1 >= 0:
                        if best_f2 < 0:
                            for j in range(cnt):
                                v = XT[best_f1, idx[a + j]]
                                vals[j] = v
                                raw_vals[j] = v
                        else:
                            for j in range(cnt):
                                r = idx[a + j]
                                v = XT[best_f1, r] - XT[best_f2, r]
                                vals[j] = v
                                raw_vals[j] = v
                        ok_full, best_thr, _ = _select_threshold_nan(vals, cnt, qs[t, node])

                    if ok_full:
                        feature1[t, node] = best_f1
                        feature2[t, node] = best_f2
                        threshold[t, node] = best_thr
                        nan_right[t, node] = best_nr
                        p = a
                        for j in range(cnt):
                            rv = raw_vals[j]
                            go_left = (best_nr == 0) if np.isnan(rv) else (rv <= best_thr)
                            if go_left:
                                idx[p], idx[a + j] = idx[a + j], idx[p]
                                p += 1
                        for j in range(p - 1, a, -1):
                            rng_state ^= rng_state << np.uint32(13)
                            rng_state ^= rng_state >> np.uint32(17)
                            rng_state ^= rng_state << np.uint32(5)
                            r = a + int(rng_state % np.uint32(j - a + 1))
                            idx[j], idx[r] = idx[r], idx[j]
                        for j in range(b - 1, p, -1):
                            rng_state ^= rng_state << np.uint32(13)
                            rng_state ^= rng_state >> np.uint32(17)
                            rng_state ^= rng_state << np.uint32(5)
                            r = p + int(rng_state % np.uint32(j - p + 1))
                            idx[j], idx[r] = idx[r], idx[j]
                        split = p
                    else:
                        split = b
                lo[2 * node + 1], hi[2 * node + 1] = a, split
                lo[2 * node + 2], hi[2 * node + 2] = split, b

    @numba.njit(**_NJIT_PAR)
    def _fit_oblivious_nb(XT, boot, perms, qs, depth, feature, threshold):
        T, m = boot.shape
        d = perms.shape[1]
        for t in numba.prange(T):
            idx = boot[t]
            vals = np.empty(m, dtype=XT.dtype)
            chosen = 0
            for k in range(d):
                f = perms[t, k]
                for j in range(m):
                    vals[j] = XT[f, idx[j]]
                ok, thr, _ = _select_threshold_nan(vals, m, qs[t, chosen])
                if ok:
                    feature[t, chosen] = f
                    threshold[t, chosen] = thr
                    chosen += 1
                    if chosen == depth:
                        break
            if chosen < depth:
                f0 = feature[t, 0]
                for lvl in range(chosen, depth):
                    feature[t, lvl] = f0
                    threshold[t, lvl] = np.float32(1e30)

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


def _validate_X(X):
    """Ensure X is a 2D float32 array and raise a clear ValueError if +/-inf is present."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array X, got shape {X.shape}")
    if np.isinf(X).any():
        raise ValueError("Input X contains infinity (inf or -inf). Please replace or clip infinite values.")
    return X


class LeafSGD:
    """Lock-free parallel Hogwild! SGD on leaf indices.

    eta="auto"            tries eta in (0.15, 0.5, 1.5, 3.0) for 2 epochs each on validation,
                          then continues from the best.
    init="newton"         starts every leaf weight at its one-step Newton value (as gradient
                          boosting sets leaf values): summed residual of the leaf's training
                          rows / (rows * p(1-p) + lambda), divided by the number of trees; the
                          intercept starts at the class log-priors. Default: zeros.
    leaf_l2               > 0 switches to count-aware (ridge-style) shrinkage: every leaf gets
                          the same total penalty per epoch, so leaves with few rows are pulled
                          towards zero hardest, like lambda in gradient boosting's leaf values.
                          0 keeps the old per-visit decay with strength `l2`.
    adagrad=True          per-weight step sizes (larger for rarely visited leaves); starts out
                          equivalent to plain SGD, then each weight's step shrinks with its own
                          gradient history instead of the global schedule.
    stop_metric           "accuracy" (default) or "logloss" for early stopping / tail averaging.
    """

    def __init__(self, loss="softmax", eta=0.5, l2=1e-6, drop_p=0.0, tail_avg=False,
                 max_epochs=20, patience=3, device="cpu", random_state=0,
                 init=None, leaf_l2=0.0, adagrad=False, stop_metric="accuracy"):
        self.loss, self.eta, self.l2 = loss, eta, l2
        self.drop_p, self.tail_avg = drop_p, tail_avg
        self.max_epochs, self.patience = max_epochs, patience
        self.device, self.random_state = device, random_state
        self.init, self.leaf_l2, self.adagrad, self.stop_metric = init, leaf_l2, adagrad, stop_metric

    # -- validation score: higher is better ------------------------------------------------
    def _val_score(self, s_va, y_val):
        if self.stop_metric == "accuracy":
            if s_va.ndim == 1:
                return float((self.classes_[(s_va > 0.0).astype(np.intp)] == y_val).mean())
            return float((self.classes_[s_va.argmax(1)] == y_val).mean())
        if self.stop_metric != "logloss":
            raise ValueError(f"unknown stop_metric: {self.stop_metric!r}")
        yv, ok = self._yv_idx, self._yv_ok
        s = s_va.astype(np.float64)
        if s.ndim == 1:                                  # binary logit
            y1 = (yv == 1)[ok]
            s = s[ok]
            ll = np.logaddexp(0.0, -s) * y1 + np.logaddexp(0.0, s) * (~y1)
            return -float(ll.mean())
        s = s[ok]
        m = s.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(s - m).sum(1))
        return -float((lse - s[np.arange(len(s)), yv[ok]]).mean())

    def _run_epochs(self, leaves, yi, leaves_val, y_val, X, X_val, O, O_va,
                    eta_val, max_ep, pat, W_init=None, b_init=None, V_init=None, start_ep=0):
        n, T = leaves.shape
        C = len(self.classes_)
        n_leaves = self.n_leaves_
        n_rows = T * n_leaves
        l2_f32 = np.float32(self.l2_leaf_)
        l2_raw = np.float32(self.l2)
        drop_f32 = float(self.drop_p)
        rng = np.random.default_rng(self.random_state)
        ada = bool(self.adagrad)
        # Adagrad: the accumulator starts at g2_0 (a typical epoch's squared gradients) and the
        # rate is scaled by sqrt(g2_0), so the first steps match plain SGD; afterwards each
        # weight's step shrinks with its own gradient history (the global decay is switched off).
        g2_0 = np.float32(0.25 * max(1.0, n / n_leaves))
        ada_scale = float(np.sqrt(g2_0)) if ada else 1.0

        if C == 2:
            W1 = np.zeros(n_rows, dtype=np.float32) if W_init is None else W_init.copy()
            b1 = np.zeros(1, dtype=np.float32) if b_init is None else b_init.copy()
            G2 = np.full(n_rows if ada else 1, g2_0, dtype=np.float32)
            V1 = None
            if X is not None:
                V1 = np.zeros(X.shape[1], dtype=np.float32) if V_init is None else V_init.copy()
                sq = max(1.0, float((X ** 2).sum(1).mean()))
            best, best_state, waited = -np.inf, None, 0
            top_snapshots = []
            history = []

            for epoch in range(start_ep, max_ep):
                decay = 1.0 if ada else 1.0 + 0.5 * epoch
                lr = np.float32(eta_val / T / decay * ada_scale)
                order = rng.permutation(n).astype(np.int32)
                seed = int(rng.integers(1, 2**31 - 1)) if drop_f32 > 0.0 else 1
                if X is None:
                    _sgd_epoch_bin_nb(leaves, yi, order, self.depth_, W1, b1, O,
                                      lr, l2_f32, self.loss == "softmax", drop_f32, seed,
                                      self.l2w_, G2, ada)
                else:
                    _sgd_epoch_bin_raw_nb(leaves, X, yi, order, self.depth_, W1, V1, b1, O,
                                          np.float32(0.5 * lr), np.float32(0.5 * eta_val / sq / (1.0 + 0.5 * epoch)),
                                          l2_f32, self.loss == "softmax", drop_f32, seed,
                                          l2_raw, self.l2w_, G2, ada)
                s_va = _lookup_predict_bin_nb(leaves_val, self.depth_, W1, b1[0])
                if V1 is not None:
                    s_va += X_val @ V1
                if O_va is not None:
                    s_va += O_va
                acc = self._val_score(s_va, y_val)
                history.append(acc)
                snap = (W1.copy(), b1.copy(), None if V1 is None else V1.copy())
                if self.tail_avg:
                    top_snapshots.append((acc, epoch, snap))
                    top_snapshots.sort(key=lambda item: (item[0], item[1]), reverse=True)
                    if len(top_snapshots) > 3:
                        top_snapshots.pop()
                if acc > best:
                    best, best_state, waited = acc, snap, 0
                else:
                    waited += 1
                    if waited >= pat:
                        break

            W1, b1, V1 = best_state
            if self.tail_avg and len(top_snapshots) >= 2:
                k = len(top_snapshots)
                W1_avg = sum(s[2][0] for s in top_snapshots) / np.float32(k)
                b1_avg = sum(s[2][1] for s in top_snapshots) / np.float32(k)
                V1_avg = None if V1 is None else (sum(s[2][2] for s in top_snapshots) / np.float32(k))
                s_va = _lookup_predict_bin_nb(leaves_val, self.depth_, W1_avg, b1_avg[0])
                if V1_avg is not None:
                    s_va += X_val @ V1_avg
                if O_va is not None:
                    s_va += O_va
                acc_avg = self._val_score(s_va, y_val)
                if acc_avg >= best:
                    best = acc_avg
                    W1, b1, V1 = W1_avg, b1_avg, V1_avg
            return best, (W1, b1, V1), history

        # Multiclass C >= 3
        W = np.zeros((n_rows, C), dtype=np.float32) if W_init is None else W_init.copy()
        b = np.zeros(C, dtype=np.float32) if b_init is None else b_init.copy()
        G2 = np.full((n_rows, C) if ada else (1, 1), g2_0, dtype=np.float32)
        V = None
        if X is not None:
            V = np.zeros((X.shape[1], C), dtype=np.float32) if V_init is None else V_init.copy()
            sq = max(1.0, float((X ** 2).sum(1).mean()))
        best, best_state, waited = -np.inf, None, 0
        top_snapshots = []
        history = []

        for epoch in range(start_ep, max_ep):
            decay = 1.0 if ada else 1.0 + 0.5 * epoch
            lr = np.float32(eta_val / T / decay * ada_scale)
            order = rng.permutation(n).astype(np.int32)
            seed = int(rng.integers(1, 2**31 - 1)) if drop_f32 > 0.0 else 1
            if X is None:
                _sgd_epoch_nb(leaves, yi, order, self.depth_, W, b, O,
                              lr, l2_f32, self.loss == "softmax", drop_f32, seed,
                              self.l2w_, G2, ada)
            else:
                _sgd_epoch_raw_nb(leaves, X, yi, order, self.depth_, W, V, b, O,
                                  np.float32(0.5 * lr), np.float32(0.5 * eta_val / sq / (1.0 + 0.5 * epoch)),
                                  l2_f32, self.loss == "softmax", drop_f32, seed,
                                  l2_raw, self.l2w_, G2, ada)
            s_va = _lookup_predict_nb(leaves_val, self.depth_, W, b)
            if V is not None:
                s_va += X_val @ V
            if O_va is not None:
                s_va += O_va
            acc = self._val_score(s_va, y_val)
            history.append(acc)
            snap = (W.copy(), b.copy(), None if V is None else V.copy())
            if self.tail_avg:
                top_snapshots.append((acc, epoch, snap))
                top_snapshots.sort(key=lambda item: (item[0], item[1]), reverse=True)
                if len(top_snapshots) > 3:
                    top_snapshots.pop()
            if acc > best:
                best, best_state, waited = acc, snap, 0
            else:
                waited += 1
                if waited >= pat:
                    break

        W, b, V = best_state
        if self.tail_avg and len(top_snapshots) >= 2:
            k = len(top_snapshots)
            W_avg = sum(s[2][0] for s in top_snapshots) / np.float32(k)
            b_avg = sum(s[2][1] for s in top_snapshots) / np.float32(k)
            V_avg = None if V is None else (sum(s[2][2] for s in top_snapshots) / np.float32(k))
            s_va = _lookup_predict_nb(leaves_val, self.depth_, W_avg, b_avg)
            if V_avg is not None:
                s_va += X_val @ V_avg
            if O_va is not None:
                s_va += O_va
            acc_avg = self._val_score(s_va, y_val)
            if acc_avg >= best:
                best = acc_avg
                W, b, V = W_avg, b_avg, V_avg
        return best, (W, b, V), history

    def _initial_weights(self, leaves, yi, has_offset):
        """Leaf counts (for count-aware L2) and, with init='newton', starting weights."""
        n, T = leaves.shape
        C = len(self.classes_)
        n_rows = T * self.n_leaves_
        prior = np.bincount(yi, minlength=C).astype(np.float64) / len(yi)
        prior = np.clip(prior, 1e-4, 1 - 1e-4)
        if C == 2:
            G = (yi == 1).astype(np.float32)[:, None] - np.float32(prior[1])
        else:
            G = (np.eye(C, dtype=np.float32)[yi] - prior.astype(np.float32))
        cnt, S = _leaf_stats_nb(leaves, np.int32(self.depth_), np.ascontiguousarray(G), n_rows)
        self.leaf_counts_ = cnt
        if self.leaf_l2 > 0:
            self.l2w_ = (1.0 / np.maximum(cnt, 1.0)).astype(np.float32)
            self.l2_leaf_ = float(self.leaf_l2)
        else:
            self.l2w_ = np.ones(n_rows, dtype=np.float32)
            self.l2_leaf_ = float(self.l2)
        if self.init is None:
            return None, None
        if self.init != "newton":
            raise ValueError(f"unknown init: {self.init!r}")
        lam = max(float(self.leaf_l2), 1.0)
        if C == 2:
            h = prior[1] * (1 - prior[1])
            W0 = (S[:, 0] / (cnt * h + lam) / T).astype(np.float32)
            b0 = np.array([np.log(prior[1] / prior[0])], dtype=np.float32)
        else:
            h = (prior * (1 - prior)).astype(np.float32)
            W0 = (S / (cnt[:, None] * h + lam) / T).astype(np.float32)
            b0 = np.log(prior).astype(np.float32)
        if has_offset:                                   # earlier stages already set the level
            b0 = np.zeros_like(b0)
        return np.ascontiguousarray(W0), b0

    def fit(self, leaves, y, leaves_val, y_val, n_leaves, X=None, X_val=None,
            offset=None, offset_val=None, classes=None):
        if not HAVE_NUMBA:
            raise ImportError("LeafSGD needs numba (pip install numba)")
        if classes is None:
            self.classes_, yi = np.unique(y, return_inverse=True)
        else:
            self.classes_ = np.asarray(classes)
            yi = np.searchsorted(self.classes_, y)
        yi = yi.astype(np.int64)
        y_val = np.asarray(y_val)
        pos = np.clip(np.searchsorted(self.classes_, y_val), 0, len(self.classes_) - 1)
        self._yv_ok = self.classes_[pos] == y_val
        self._yv_idx = pos.astype(np.int64)
        leaves = np.ascontiguousarray(leaves, dtype=np.int32)
        leaves_val = np.ascontiguousarray(leaves_val, dtype=np.int32)

        n, _ = leaves.shape
        C = len(self.classes_)
        self.n_leaves_ = n_leaves
        self.depth_ = int(round(np.log2(n_leaves)))
        self.is_binary_1d_ = (C == 2)

        if X is not None:
            X = np.ascontiguousarray(X, dtype=np.float32)
            X_val = np.ascontiguousarray(X_val, dtype=np.float32)

        if C == 2:
            O = np.zeros(n, dtype=np.float32) if offset is None else np.ascontiguousarray(
                offset[:, 1] - offset[:, 0] if offset.ndim == 2 else offset, dtype=np.float32
            )
            O_va = None if offset_val is None else np.ascontiguousarray(
                offset_val[:, 1] - offset_val[:, 0] if offset_val.ndim == 2 else offset_val, dtype=np.float32
            )
        else:
            O = np.zeros((n, C), dtype=np.float32) if offset is None else np.ascontiguousarray(offset, dtype=np.float32)
            O_va = None if offset_val is None else np.ascontiguousarray(offset_val, dtype=np.float32)

        W0, b0 = self._initial_weights(leaves, yi, offset is not None)

        if self.eta == "auto":
            # Validation-scored LR grid across noisy (0.15) to clean (3.0) regimes (2 fast Hogwild epochs each)
            best_probe_acc = -np.inf
            best_eta = 0.5
            best_init = None
            for cand_eta in (0.15, 0.5, 1.5, 3.0):
                acc_c, state_c, _ = self._run_epochs(
                    leaves, yi, leaves_val, y_val, X, X_val, O, O_va,
                    eta_val=cand_eta, max_ep=2, pat=2, W_init=W0, b_init=b0
                )
                if acc_c > best_probe_acc:
                    best_probe_acc = acc_c
                    best_eta = cand_eta
                    best_init = state_c
            self.eta_ = best_eta
            _, final_state, self.history_ = self._run_epochs(
                leaves, yi, leaves_val, y_val, X, X_val, O, O_va,
                eta_val=self.eta_, max_ep=self.max_epochs, pat=self.patience,
                W_init=best_init[0], b_init=best_init[1], V_init=best_init[2], start_ep=2
            )
        else:
            self.eta_ = float(self.eta)
            _, final_state, self.history_ = self._run_epochs(
                leaves, yi, leaves_val, y_val, X, X_val, O, O_va,
                eta_val=self.eta_, max_ep=self.max_epochs, pat=self.patience,
                W_init=W0, b_init=b0
            )

        if C == 2:
            self.W1_, self.b1_, self.V1_ = final_state
        else:
            self.W_, self.b_, self.V_ = final_state
        self.n_epochs_ = len(self.history_)
        return self

    def decision_function(self, leaves, X=None, offset=None):
        leaves = np.ascontiguousarray(leaves, dtype=np.int32)
        if getattr(self, "is_binary_1d_", False):
            s1 = _lookup_predict_bin_nb(leaves, self.depth_, self.W1_, self.b1_[0])
            if self.V1_ is not None:
                if X is None:
                    raise ValueError("this model was trained with raw features; pass X")
                s1 += np.asarray(X, dtype=np.float32) @ self.V1_
            out = np.empty((len(leaves), 2), dtype=np.float32)
            out[:, 0] = -0.5 * s1
            out[:, 1] = 0.5 * s1
            if offset is not None:
                out += offset
            return out

        s = _lookup_predict_nb(leaves, self.depth_, self.W_, self.b_)
        if self.V_ is not None:
            if X is None:
                raise ValueError("this model was trained with raw features; pass X")
            s += np.asarray(X, dtype=np.float32) @ self.V_
        if offset is not None:
            s += offset
        return s

    def predict(self, leaves, X=None, offset=None):
        if getattr(self, "is_binary_1d_", False) and offset is None:
            leaves = np.ascontiguousarray(leaves, dtype=np.int32)
            s1 = _lookup_predict_bin_nb(leaves, self.depth_, self.W1_, self.b1_[0])
            if self.V1_ is not None:
                s1 += np.asarray(X, dtype=np.float32) @ self.V1_
            return self.classes_[(s1 > 0.0).astype(np.intp)]
        return self.classes_[self.decision_function(leaves, X, offset).argmax(1)]


def _to_csr(leaves, n_leaves):
    n, T = leaves.shape
    cols = (leaves + np.arange(T, dtype=np.int32) * n_leaves).ravel()
    return sp.csr_matrix((np.ones(n * T, dtype=np.float32), cols, np.arange(0, n * T + 1, T)),
                         shape=(n, T * n_leaves))


class _Base(TransformerMixin, BaseEstimator):
    def __init__(self, n_trees=64, depth=7, max_samples=4096, random_state=None):
        self.n_trees, self.depth = n_trees, depth
        self.max_samples, self.random_state = max_samples, random_state

    @property
    def n_features_out(self):
        return self.n_trees * (1 << self.depth)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def _sample(self, X, rng):
        m = min(self.max_samples, len(X))
        return X[rng.integers(0, len(X), m)]

    def leaf_balance(self, X):
        leaves = self.apply(X)
        out = []
        for t in range(leaves.shape[1]):
            _, c = np.unique(leaves[:, t], return_counts=True)
            p = c / c.sum()
            out.append(np.exp(-(p * np.log(p)).sum()) / (1 << self.depth))
        return float(np.mean(out))

    def transform(self, X):
        X = _validate_X(X)
        return _to_csr(self.apply(X), 1 << self.depth)

    def predict_linear(self, X, W, b):
        leaves = self.apply(X)
        W = np.asarray(W, dtype=np.float32)
        W = np.ascontiguousarray(W.reshape(len(W), -1))
        b = np.atleast_1d(np.asarray(b, dtype=np.float32)).ravel()
        if HAVE_NUMBA:
            return _lookup_predict_nb(leaves, self.depth, W, b)
        rows = leaves + np.arange(self.n_trees) * (1 << self.depth)
        return W[rows].sum(1) + b


class MedianTreeEmbedder(_Base):
    """Full binary median trees with native missing-value (NaN) routing and optional
    pairwise contrastive difference (x_i - x_j) splits (`oblique_frac > 0`)."""

    def __init__(self, n_trees=64, depth=7, max_samples=4096, feature_weights=None,
                 quantile_jitter=0.0, oblique_frac=0.0, n_candidates=None,
                 score_rows=256, device="cpu", random_state=None):
        self.n_trees, self.depth, self.max_samples = n_trees, depth, max_samples
        self.feature_weights, self.quantile_jitter = feature_weights, quantile_jitter
        self.oblique_frac, self.device = oblique_frac, device
        self.n_candidates, self.score_rows, self.random_state = n_candidates, score_rows, random_state

    def _candidates(self, rng, d, shape):
        if self.feature_weights is None:
            c1 = rng.integers(0, d, shape, dtype=np.int32)
        else:
            p = np.asarray(self.feature_weights, dtype=np.float64)
            c1 = rng.choice(d, size=shape, p=p / p.sum()).astype(np.int32)

        if self.oblique_frac <= 0.0 or d < 2:
            c2 = np.full(shape, -1, dtype=np.int32)
        else:
            if self.feature_weights is None:
                c2 = rng.integers(0, d, shape, dtype=np.int32)
            else:
                c2 = rng.choice(d, size=shape, p=p / p.sum()).astype(np.int32)
            mask = (rng.random(shape) >= self.oblique_frac) | (c1 == c2)
            c2[mask] = -1
        return c1, c2

    def fit(self, X, y=None, sample_weight=None, targets=None):
        X = _validate_X(X)
        rng = np.random.default_rng(self.random_state)
        D, d = self.depth, X.shape[1]
        n_internal = (1 << D) - 1
        self.feature_ = np.zeros((self.n_trees, n_internal), dtype=np.int32)
        self.feature2_ = np.full((self.n_trees, n_internal), -1, dtype=np.int32)
        self.threshold_ = np.full((self.n_trees, n_internal), 1e30, dtype=np.float32)
        self.nan_right_ = np.zeros((self.n_trees, n_internal), dtype=np.int8)

        qs = (0.5 + rng.uniform(-self.quantile_jitter, self.quantile_jitter,
                                (self.n_trees, n_internal))).astype(np.float32)
        m = min(self.max_samples, len(X))
        if sample_weight is None:
            boot = rng.integers(0, len(X), (self.n_trees, m), dtype=np.int32)
        else:
            p = np.asarray(sample_weight, dtype=np.float64)
            boot = rng.choice(len(X), size=(self.n_trees, m), p=p / p.sum()).astype(np.int32)

        if targets is not None:
            if not HAVE_NUMBA:
                raise ImportError("supervised splits need numba (pip install numba)")
            k = self.n_candidates or 6
            cand1, cand2 = self._candidates(rng, d, (self.n_trees, n_internal, k))
            base_seed = int(rng.integers(1, 2**30))
            _fit_sup_nb(np.ascontiguousarray(X.T), np.ascontiguousarray(targets, dtype=np.float32),
                        boot, cand1, cand2, qs, D, int(self.score_rows), base_seed,
                        self.feature_, self.feature2_, self.threshold_, self.nan_right_)
        elif HAVE_NUMBA:
            cand1, cand2 = self._candidates(rng, d, (self.n_trees, n_internal, 8))
            _fit_full_nb(np.ascontiguousarray(X.T), boot, cand1, cand2, qs, D,
                         self.feature_, self.feature2_, self.threshold_, self.nan_right_)
        else:
            raise ImportError("numba required")
        return self

    def apply(self, X):
        X = _validate_X(X)
        if HAVE_NUMBA:
            return _apply_full_nb(X, self.feature_, self.feature2_, self.threshold_,
                                  self.nan_right_, self.depth)
        raise ImportError("numba required")


class ObliviousMedianEmbedder(_Base):
    def fit(self, X, y=None):
        X = _validate_X(X)
        rng = np.random.default_rng(self.random_state)
        T, D, d = self.n_trees, self.depth, X.shape[1]
        self.feature_ = np.zeros((T, D), dtype=np.int32)
        self.threshold_ = np.full((T, D), 1e30, dtype=np.float32)
        m = min(self.max_samples, len(X))
        boot = rng.integers(0, len(X), (T, m), dtype=np.int32)
        qs = np.full((T, D), 0.5, dtype=np.float32)
        if HAVE_NUMBA:
            perms = np.empty((T, d), dtype=np.int32)
            for t in range(T):
                perms[t] = rng.permutation(d)
            _fit_oblivious_nb(np.ascontiguousarray(X.T), boot, perms, qs, D, self.feature_, self.threshold_)
        self._F = self.feature_.ravel()
        self._thr = self.threshold_.ravel()
        self._weights = 1 << np.arange(D, dtype=np.int32)
        return self

    def apply(self, X):
        X = _validate_X(X)
        if HAVE_NUMBA:
            return _apply_oblivious_nb(X, self.feature_, self.threshold_)
        T, D = self.n_trees, self.depth
        out = np.empty((len(X), T), dtype=np.int32)
        for s in range(0, len(X), _CHUNK):
            bits = (X[s:s + _CHUNK][:, self._F] > self._thr).reshape(-1, T, D)
            out[s:s + _CHUNK] = bits @ self._weights
        return out


class _QuantileBinner:
    """Maps each feature to its quantile rank in [0, 1] on a grid of at most n_bins levels
    (like gradient boosting's histogram binning). NaNs stay NaN. Median splits are unchanged
    by this, but oblique splits (x_i - x_j) then compare ranks instead of raw values, so
    outliers and skew no longer dominate them, and thresholds can only fall between bins."""

    def __init__(self, n_bins=255, max_rows=200000, random_state=0):
        self.n_bins, self.max_rows, self.random_state = n_bins, max_rows, random_state

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        sub = X if len(X) <= self.max_rows else X[rng.choice(len(X), self.max_rows, replace=False)]
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j in range(X.shape[1]):
                col = sub[:, j]
                col = col[~np.isnan(col)]
                self.edges_.append(np.unique(np.quantile(col, qs)).astype(np.float32)
                                   if col.size else np.zeros(0, np.float32))
        return self

    def transform(self, X):
        out = np.empty(X.shape, dtype=np.float32)
        for j, e in enumerate(self.edges_):
            col = X[:, j]
            r = np.searchsorted(e, col, side="right").astype(np.float32) / max(1, len(e))
            out[:, j] = np.where(np.isnan(col), np.float32(np.nan), r)
        return np.ascontiguousarray(out)


class _CategoryEncoder:
    """Replaces integer-coded categorical columns by smoothed class frequencies of each
    category (1 column for binary targets, one per class otherwise), the counterpart of
    gradient boosting ordering categories by their gradient. Training rows are encoded
    out-of-fold so a row's own label never leaks into its encoding. NaN = its own category;
    unseen categories get the prior."""

    def __init__(self, columns, smoothing=10.0, n_folds=5, random_state=0):
        self.columns, self.smoothing = list(columns), smoothing
        self.n_folds, self.random_state = n_folds, random_state

    def _stats(self, codes, yi):
        cats = np.unique(codes[~np.isnan(codes)])
        idx = np.where(np.isnan(codes), len(cats), np.searchsorted(cats, codes))
        counts = np.zeros((len(cats) + 1, self.C_))
        np.add.at(counts, (idx, yi), 1.0)
        n = counts.sum(1, keepdims=True)
        enc = (counts + self.smoothing * self.prior_) / (n + self.smoothing)
        return cats, enc

    def _lookup(self, codes, cats, enc):
        pos = np.clip(np.searchsorted(cats, codes), 0, max(len(cats) - 1, 0))
        known = (~np.isnan(codes)) & (len(cats) > 0)
        if len(cats):
            known &= cats[pos] == codes
        out = np.tile(self.prior_, (len(codes), 1))
        out[known] = enc[pos[known]]
        out[np.isnan(codes)] = enc[len(cats)]
        return out[:, self.keep_]

    def fit_transform_oof(self, X, yi, C):
        self.C_ = C
        self.prior_ = np.bincount(yi, minlength=C) / len(yi)
        self.keep_ = [1] if C == 2 else list(range(C))
        self.other_ = [j for j in range(X.shape[1]) if j not in set(self.columns)]
        folds = np.random.default_rng(self.random_state).integers(0, self.n_folds, len(yi))
        parts = []
        self.tables_ = []
        for j in self.columns:
            col = X[:, j].astype(np.float64)
            oof = np.empty((len(col), len(self.keep_)))
            for k in range(self.n_folds):
                cats, enc = self._stats(col[folds != k], yi[folds != k])
                oof[folds == k] = self._lookup(col[folds == k], cats, enc)
            parts.append(oof)
            self.tables_.append(self._stats(col, yi))
        return np.ascontiguousarray(np.hstack([X[:, self.other_]] + parts), dtype=np.float32)

    def transform(self, X):
        parts = [self._lookup(X[:, j].astype(np.float64), *tab)
                 for j, tab in zip(self.columns, self.tables_)]
        return np.ascontiguousarray(np.hstack([X[:, self.other_]] + parts), dtype=np.float32)


class _MixedDepthEmbedder:
    """Several MedianTreeEmbedders of different depths used as one forest. Leaves of every
    tree index into slots of the deepest tree's size, so one LeafSGD head covers them all."""

    def __init__(self, embedders):
        self.embedders = embedders
        self.depth = max(e.depth for e in embedders)
        self.n_trees = sum(e.n_trees for e in embedders)

    def apply(self, X):
        return np.ascontiguousarray(np.hstack([e.apply(X) for e in self.embedders]))


def _softmax(S):
    E = np.exp(S - S.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


class MedianTreeClassifier(ClassifierMixin, BaseEstimator):
    """Median-tree classifier: a forest of quantile-split trees, one softmax head on its leaves.

    Defaults are the configuration used in the benchmarks: 1024 trees of depth 7, supervised
    feature choice (best of 6 candidates, half of them differences of two features), cuts at
    the node median, and a softmax head trained by parallel SGD with tree dropout (0.2),
    an automatic learning-rate search, early stopping and tail averaging.

    Parameters most worth knowing
      n_trees         number of trees (accuracy is flat above ~1024; fewer trains faster)
      depth           tree depth; 128 leaves at the default 7
      raw_features    also give the head the (standardized) raw features; helps on embeddings
      probe           pick depth, oblique_frac, dropout and raw_features on a small probe forest
      random_state    seed for reproducible forests and training

    ``fit(X, y, X_val=None, y_val=None)``: without a validation set, 10% of the training rows
    are held out for early stopping. X may contain NaN (routed natively in every split).

    Experimental options (off by default, not part of the benchmarked configuration):
      binning, init="newton", leaf_l2, adagrad, stop_metric="logloss", categorical_features,
      depth=(6, 7, 8) mixed depths, n_stages > 1, oblivious=True.
    """

    def __init__(self, n_trees=1024, depth=7, oblivious=False, max_samples=4096,
                 raw_features=False, feature_weighting=None, feature_mix=0.2,
                 supervised=True, n_candidates=6, score_rows=256,
                 oblique_frac=0.5, n_stages=1, quantile_jitter=0.0,
                 validation_fraction=0.1, eta="auto", l2=1e-6, drop_p=0.2, tail_avg=True,
                 probe=False, device="cpu", max_epochs=25, patience=4, random_state=None,
                 binning=False, n_bins=255, init=None, leaf_l2=0.0, adagrad=False,
                 stop_metric="accuracy", categorical_features=None, cat_smoothing=10.0):
        self.binning, self.n_bins, self.init, self.leaf_l2 = binning, n_bins, init, leaf_l2
        self.adagrad, self.stop_metric = adagrad, stop_metric
        self.categorical_features, self.cat_smoothing = categorical_features, cat_smoothing
        self.n_trees, self.depth, self.oblivious, self.max_samples = n_trees, depth, oblivious, max_samples
        self.raw_features, self.feature_weighting = raw_features, feature_weighting
        self.feature_mix, self.supervised = feature_mix, supervised
        self.n_candidates, self.score_rows = n_candidates, score_rows
        self.oblique_frac, self.n_stages = oblique_frac, n_stages
        self.quantile_jitter = quantile_jitter
        self.validation_fraction, self.eta, self.l2 = validation_fraction, eta, l2
        self.drop_p, self.tail_avg = drop_p, tail_avg
        self.probe, self.device = probe, device
        self.max_epochs, self.patience, self.random_state = max_epochs, patience, random_state

    def _fit_scaler(self, X_tr):
        """NaN-safe centering and scaling; keeps NaNs intact in X_tr_s for native tree routing,
        and provides median-imputed array `R` when dense linear `raw_features` are used."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(X_tr, axis=0)
            sd = np.nanstd(X_tr, axis=0)
            c_min = np.nanmin(X_tr, axis=0)
            c_max = np.nanmax(X_tr, axis=0)
        self.mean_ = np.where(np.isnan(mean), np.float32(0.0), mean).astype(np.float32)
        sd = np.where(np.isnan(sd) | (sd == 0.0), np.float32(1.0), sd).astype(np.float32)
        span = np.where(np.isnan(c_max - c_min), np.float32(0.0), c_max - c_min)
        # Prevent rare binary 0/1 columns from blowing up when divided by tiny std
        self.scale_ = np.where((span > 0.0) & (span <= 1.0),
                               np.maximum(sd, np.float32(0.35)), sd).astype(np.float32)

    def _prep_X(self, X, encoded=None):
        """Returns (tree inputs, dense inputs for the raw linear term). `encoded` passes
        training rows that were already categorical-encoded out-of-fold."""
        if encoded is not None:
            X = encoded
        else:
            X = _validate_X(X)
            if getattr(self, "cat_encoder_", None) is not None:
                X = self.cat_encoder_.transform(X)
        if not self.need_scale_:
            X_t, R = X, X
        else:
            X_s = np.ascontiguousarray((X - self.mean_) / self.scale_, dtype=np.float32)
            if np.isnan(X_s).any():
                # Keep NaNs in X_s for native tree routing; impute 0.0 (the mean) for dense linear V
                R = np.ascontiguousarray(np.where(np.isnan(X_s), np.float32(0.0), X_s), dtype=np.float32)
            else:
                R = X_s
            X_t = X_s
        if getattr(self, "binner_", None) is not None:
            X_t = self.binner_.transform(X)
        return X_t, R

    def _head_extra(self):
        return dict(init=self.init, leaf_l2=self.leaf_l2, adagrad=self.adagrad,
                    stop_metric=self.stop_metric)

    def _run_micro_probe(self, X_tr_s, R_tr_s, y_tr, X_va_s, R_va_s, y_va, Y_tr, weights, rng):
        """Dataset-agnostic validation micro-probe (no hardcoded dimension rules):
          1. Sets tree depth from sample support per leaf: clip(floor(log2(n / 150)), 6, 8)
          2. Tests oblique_frac in (0.0, 0.5) on a 64-tree micro-forest
          3. Tests raw_features in (False, True) on the winning micro-forest
          4. Sets n_trees from sub-forest scaling gain (32 -> 64 trees)
        """
        n, d = X_tr_s.shape
        C = Y_tr.shape[1]
        max_trees = 1024 if self.n_trees == "auto" else int(self.n_trees)

        # 1. Principled leaf-occupancy rule: target ~150 training samples per leaf
        if self.depth == "auto" or self.probe:
            probed_depth = int(np.clip(int(np.log2(max(n, 64) / 150.0)), 6, 8))
        else:
            probed_depth = int(self.depth)

        n_sub = min(n, 4096)
        sub_idx = rng.choice(n, size=n_sub, replace=False) if n > n_sub else np.arange(n)
        X_sub = np.ascontiguousarray(X_tr_s[sub_idx])
        R_sub = np.ascontiguousarray(R_tr_s[sub_idx])
        y_sub = y_tr[sub_idx]
        targets_sub = np.ascontiguousarray(Y_tr[sub_idx] - Y_tr[sub_idx].mean(0, keepdims=True))

        n_va_sub = min(len(y_va), 2048)
        X_va_sub = np.ascontiguousarray(X_va_s[:n_va_sub])
        R_va_sub = np.ascontiguousarray(R_va_s[:n_va_sub])
        y_va_sub = y_va[:n_va_sub]

        cand_k = int(np.clip(int(round(np.sqrt(d))), 4, 8))
        probe_d = min(probed_depth, 6)
        n_leaves_p = 1 << probe_d

        # 2. Test axis-aligned (0.0) vs oblique (0.5) on 64 micro-trees
        best_oblique = 0.5
        best_score = -1.0
        best_L_tr, best_L_va, best_head = None, None, None

        for ob_frac in (0.0, 0.5):
            emb_p = MedianTreeEmbedder(
                n_trees=64, depth=probe_d, max_samples=min(2048, n_sub),
                feature_weights=weights, oblique_frac=ob_frac,
                n_candidates=cand_k, score_rows=128, random_state=0
            ).fit(X_sub, targets=targets_sub)
            L_p = emb_p.apply(X_sub)
            L_va_p = emb_p.apply(X_va_sub)
            head_p = LeafSGD(
                loss="softmax", eta=0.5, l2=self.l2, drop_p=0.0,
                max_epochs=3, patience=2, random_state=0
            ).fit(L_p, y_sub, L_va_p, y_va_sub, n_leaves=n_leaves_p, classes=self.classes_)
            acc_p = max(head_p.history_)
            if acc_p >= best_score:
                best_score = acc_p
                best_oblique = ob_frac
                best_L_tr, best_L_va = L_p, L_va_p
                best_head = head_p

        # 3. Test whether adding raw_features=True improves validation accuracy on the micro-forest
        use_raw = False
        if d <= 1024:
            head_raw = LeafSGD(
                loss="softmax", eta=0.5, l2=self.l2, drop_p=0.0,
                max_epochs=3, patience=2, random_state=0
            ).fit(best_L_tr, y_sub, best_L_va, y_va_sub, n_leaves=n_leaves_p,
                  X=R_sub, X_val=R_va_sub, classes=self.classes_)
            if max(head_raw.history_) > best_score + 0.002:
                use_raw = True
                best_score = max(head_raw.history_)

        # 4. Sub-forest scaling law: 32 -> 64 trees
        acc_32 = max(LeafSGD(
            loss="softmax", eta=0.5, l2=self.l2, max_epochs=3, patience=2, random_state=0
        ).fit(best_L_tr[:, :32], y_sub, best_L_va[:, :32], y_va_sub,
              n_leaves=n_leaves_p, classes=self.classes_).history_)

        marginal_gain = best_score - acc_32
        if marginal_gain < 0.003 and C == 2:
            probed_trees = min(max_trees, 512)
        else:
            probed_trees = max_trees

        tr_acc = float((best_head.predict(best_L_tr) == y_sub).mean())
        overfit_gap = tr_acc - best_score
        drop_p = 0.20 if (overfit_gap > 0.025 or C > 2) else 0.10

        return dict(
            n_trees=int(probed_trees),
            depth=int(probed_depth),
            oblique_frac=best_oblique,
            n_candidates=cand_k,
            drop_p=drop_p,
            raw_features=use_raw,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        X = _validate_X(X)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        if X_val is None:
            is_val = rng.random(len(y)) < self.validation_fraction
            if not is_val.any() or is_val.all():
                is_val = np.zeros(len(y), dtype=bool)
                is_val[rng.choice(len(y), size=max(1, len(y) // 10), replace=False)] = True
            X_tr, y_tr, X_va, y_va = X[~is_val], y[~is_val], X[is_val], y[is_val]
        else:
            X_tr, y_tr = X, y
            X_va = _validate_X(X_val)
            y_va = np.asarray(y_val)

        self.classes_, yi = np.unique(y_tr, return_inverse=True)
        C = len(self.classes_)
        self.cat_encoder_, self.binner_ = None, None
        mixed = isinstance(self.depth, (tuple, list))
        if mixed and (self.probe or self.oblivious or self.n_stages > 1 or self.n_trees == "auto"):
            raise ValueError("a tuple of depths needs probe=False, oblivious=False, n_stages=1 "
                             "and an integer n_trees")
        if C < 2:
            # Handle single-class edge case gracefully
            self.single_class_ = self.classes_[0]
            return self
        self.single_class_ = None
        Y_tr = np.eye(C, dtype=np.float32)[yi]

        X_tr_enc = None
        if self.categorical_features is not None and len(self.categorical_features):
            self.cat_encoder_ = _CategoryEncoder(self.categorical_features, self.cat_smoothing,
                                                 random_state=0 if self.random_state is None else self.random_state)
            X_tr_enc = self.cat_encoder_.fit_transform_oof(X_tr, yi, C)
        X_fit = X_tr if X_tr_enc is None else X_tr_enc

        self.need_scale_ = bool(self.raw_features or self.oblique_frac > 0.0 or self.probe)
        if self.need_scale_:
            self._fit_scaler(X_fit)
        if self.binning:
            self.binner_ = _QuantileBinner(int(self.n_bins)).fit(X_fit)

        X_tr_s, R_tr_all = self._prep_X(X_tr, encoded=X_tr_enc)
        X_va_s, R_va_all = self._prep_X(X_va)

        weights = None
        if self.feature_weighting == "f_classif":
            from sklearn.feature_selection import f_classif
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                F_stat = np.nan_to_num(f_classif(R_tr_all, y_tr)[0], nan=0.0, posinf=0.0)
            p = F_stat / F_stat.sum() if F_stat.sum() > 0 else np.full(X_tr_s.shape[1], 1.0 / X_tr_s.shape[1])
            weights = (1 - self.feature_mix) * p + self.feature_mix / X_tr_s.shape[1]
        elif self.feature_weighting is not None:
            raise ValueError(f"unknown feature_weighting: {self.feature_weighting!r}")

        if self.probe or self.n_trees == "auto" or self.depth == "auto":
            probed = self._run_micro_probe(X_tr_s, R_tr_all, y_tr, X_va_s, R_va_all, y_va, Y_tr, weights, rng)
            n_trees = probed["n_trees"] if (self.probe or self.n_trees == "auto") else int(self.n_trees)
            depth = probed["depth"] if (self.probe or self.depth == "auto") else int(self.depth)
            if self.probe:
                oblique_frac = probed["oblique_frac"]
                n_candidates = probed["n_candidates"]
                drop_p = probed["drop_p"]
                use_raw_feat = probed["raw_features"]
                supervised = True
                tail_avg = True
                eta = "auto"
                max_epochs = max(self.max_epochs, 25)
                patience = max(self.patience, 4)
            else:
                oblique_frac = self.oblique_frac
                n_candidates = self.n_candidates
                drop_p = self.drop_p
                use_raw_feat = self.raw_features
                supervised = self.supervised
                tail_avg = self.tail_avg
                eta = self.eta
                max_epochs = self.max_epochs
                patience = self.patience
            self.probed_params_ = probed
        else:
            n_trees = int(self.n_trees)
            depth = tuple(int(d) for d in self.depth) if mixed else int(self.depth)
            oblique_frac = self.oblique_frac
            n_candidates = self.n_candidates
            drop_p = self.drop_p
            use_raw_feat = self.raw_features
            supervised = self.supervised
            tail_avg = self.tail_avg
            eta = self.eta
            max_epochs = self.max_epochs
            patience = self.patience

        R_tr = R_tr_all if use_raw_feat else None
        R_va = R_va_all if use_raw_feat else None
        self.stages_ = []

        if self.oblivious:
            emb = ObliviousMedianEmbedder(
                n_trees=n_trees, depth=depth,
                max_samples=self.max_samples, random_state=self.random_state
            ).fit(X_tr_s)
            head = LeafSGD(
                loss="softmax", eta=eta, l2=self.l2, drop_p=drop_p, tail_avg=tail_avg,
                max_epochs=max_epochs, patience=patience, device=self.device, **self._head_extra(),
                random_state=self.random_state
            ).fit(emb.apply(X_tr_s), y_tr, emb.apply(X_va_s), y_va, n_leaves=1 << depth,
                  X=R_tr, X_val=R_va, classes=self.classes_)
            self.stages_.append((emb, head, use_raw_feat))
            return self

        if self.n_stages <= 1:
            targets = (Y_tr - Y_tr.mean(0, keepdims=True)) if supervised else None
            depths = list(depth) if mixed else [depth]
            per = [n_trees // len(depths)] * len(depths)
            per[-1] += n_trees - sum(per)
            embs = []
            for k, (dk, tk) in enumerate(zip(depths, per)):
                rs = self.random_state if (self.random_state is None or k == 0) else self.random_state * 7919 + k
                embs.append(MedianTreeEmbedder(
                    n_trees=tk, depth=int(dk), max_samples=self.max_samples,
                    feature_weights=weights, quantile_jitter=self.quantile_jitter,
                    oblique_frac=oblique_frac, n_candidates=n_candidates,
                    score_rows=self.score_rows, device=self.device, random_state=rs
                ).fit(X_tr_s, targets=targets))
            emb = embs[0] if len(embs) == 1 else _MixedDepthEmbedder(embs)
            depth = emb.depth
            head = LeafSGD(
                loss="softmax", eta=eta, l2=self.l2, drop_p=drop_p, tail_avg=tail_avg,
                max_epochs=max_epochs, patience=patience, device=self.device, **self._head_extra(),
                random_state=self.random_state
            ).fit(emb.apply(X_tr_s), y_tr, emb.apply(X_va_s), y_va, n_leaves=1 << depth,
                  X=R_tr, X_val=R_va, classes=self.classes_)
            self.stages_.append((emb, head, use_raw_feat))
            return self

        trees_per_stage = [n_trees // self.n_stages] * self.n_stages
        trees_per_stage[-1] += n_trees - sum(trees_per_stage)
        F_tr = np.zeros((len(y_tr), C), dtype=np.float32)
        F_va = np.zeros((len(y_va), C), dtype=np.float32)
        targets = Y_tr - Y_tr.mean(0, keepdims=True)
        sample_w = None

        for s, t_s in enumerate(trees_per_stage):
            seed = None if self.random_state is None else self.random_state * 100 + s
            emb = MedianTreeEmbedder(
                n_trees=t_s, depth=depth, max_samples=self.max_samples,
                feature_weights=weights, quantile_jitter=self.quantile_jitter,
                oblique_frac=oblique_frac, n_candidates=n_candidates,
                score_rows=self.score_rows, device=self.device, random_state=seed
            ).fit(X_tr_s, sample_weight=sample_w, targets=targets)
            L_tr = emb.apply(X_tr_s)
            L_va = emb.apply(X_va_s)
            use_raw = use_raw_feat and (s == 0)
            r_tr = R_tr if use_raw else None
            r_va = R_va if use_raw else None

            stage_eta = "auto" if eta == "auto" else float(eta) * (0.7 ** s)
            head = LeafSGD(
                loss="softmax", eta=stage_eta, l2=self.l2,
                drop_p=drop_p, tail_avg=tail_avg,
                max_epochs=max_epochs, patience=patience, device=self.device, **self._head_extra(),
                random_state=seed
            ).fit(L_tr, y_tr, L_va, y_va, n_leaves=1 << depth, X=r_tr, X_val=r_va,
                  offset=F_tr if s > 0 else None, offset_val=F_va if s > 0 else None,
                  classes=self.classes_)

            F_tr = head.decision_function(L_tr, r_tr, offset=F_tr if s > 0 else None)
            F_va = head.decision_function(L_va, r_va, offset=F_va if s > 0 else None)
            self.stages_.append((emb, head, use_raw))

            if s < self.n_stages - 1:
                P_tr = _softmax(F_tr)
                targets = Y_tr - P_tr
                if C > 2:
                    loss_row = -np.log(np.clip(P_tr[np.arange(len(y_tr)), yi], 1e-6, None))
                    sample_w = 0.5 / len(y_tr) + 0.5 * (loss_row / loss_row.sum())

        return self

    def decision_function(self, X):
        X_s, R_all = self._prep_X(X)
        if getattr(self, "single_class_", None) is not None:
            return np.zeros((len(X_s), 1), dtype=np.float32)
        F_out = None
        for emb, head, use_raw in self.stages_:
            R = R_all if use_raw else None
            F_out = head.decision_function(emb.apply(X_s), R, offset=F_out)
        return F_out

    def predict_proba(self, X):
        if getattr(self, "single_class_", None) is not None:
            return np.ones((len(X), 1), dtype=np.float32)
        return _softmax(self.decision_function(X))

    def predict(self, X):
        if getattr(self, "single_class_", None) is not None:
            return np.full(len(X), self.single_class_)
        if len(self.stages_) == 1:
            X_s, R_all = self._prep_X(X)
            emb, head, use_raw = self.stages_[0]
            return head.predict(emb.apply(X_s), R_all if use_raw else None)
        return self.classes_[self.decision_function(X).argmax(1)]

# ----------------------------------------------------------------------------- tuning

DEFAULT_GRID = [dict(depth=d, oblique_frac=o) for d in (6, 7, 8) for o in (0.0, 0.5)]


class TunedMedianTreeClassifier(ClassifierMixin, BaseEstimator):
    """Uses the method's speed for an honest validation search.

    Every configuration in `grid` (dicts of MedianTreeClassifier parameters, combined with
    `base`) is fit with `search_trees` trees and scored on the validation set; the winner is
    then refit with `n_trees` trees. With refit_full=True the winner is finally retrained on
    train + validation, keeping the learning rate and number of epochs the search chose.

        clf = TunedMedianTreeClassifier(base=dict(supervised=True, drop_p=0.2, tail_avg=True,
                                                  eta="auto"))
        clf.fit(X_train, y_train, X_val, y_val)   # or fit(X, y): holds out 15% internally
    """

    def __init__(self, grid=None, base=None, n_trees=1024, search_trees=256, refit_full=False,
                 validation_fraction=0.15, random_state=None):
        self.grid, self.base, self.n_trees, self.search_trees = grid, base, n_trees, search_trees
        self.refit_full, self.validation_fraction, self.random_state = refit_full, validation_fraction, random_state

    def fit(self, X, y, X_val=None, y_val=None):
        X = _validate_X(X)
        y = np.asarray(y)
        if X_val is None:
            rng = np.random.default_rng(self.random_state)
            is_val = rng.random(len(y)) < self.validation_fraction
            X, X_val, y, y_val = X[~is_val], X[is_val], y[~is_val], y[is_val]
        X_val = _validate_X(X_val)
        y_val = np.asarray(y_val)
        base = dict(self.base or {})
        grid = self.grid if self.grid is not None else DEFAULT_GRID
        self.search_results_ = []
        best = None
        for cfg in grid:
            params = {**base, **cfg}
            m = MedianTreeClassifier(n_trees=self.search_trees, random_state=self.random_state, **params)
            m.fit(X, y, X_val, y_val)
            score = float((m.predict(X_val) == y_val).mean())
            self.search_results_.append((cfg, score))
            if best is None or score > best[0]:
                best = (score, params)
        self.best_params_ = best[1]
        final = MedianTreeClassifier(n_trees=self.n_trees, random_state=self.random_state, **self.best_params_)
        final.fit(X, y, X_val, y_val)
        if self.refit_full:
            head = final.stages_[0][1]
            params = dict(self.best_params_, eta=head.eta_, max_epochs=max(1, head.n_epochs_),
                          patience=10 ** 6, tail_avg=False)
            X_all = np.vstack([X, X_val])
            y_all = np.concatenate([y, y_val])
            # Early stopping is off (fixed epochs), so the validation rows passed here are only
            # used for bookkeeping; they are part of the training data.
            final = MedianTreeClassifier(n_trees=self.n_trees, random_state=self.random_state, **params)
            final.fit(X_all, y_all, X_val, y_val)
        self.model_ = final
        self.classes_ = final.classes_
        return self

    def decision_function(self, X):
        return self.model_.decision_function(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)
