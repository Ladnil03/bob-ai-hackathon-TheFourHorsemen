"""Export the trained best_model to ONNX + a webapp-friendly manifest.

Exports one ``.onnx`` per ONNX-convertible base model:

* CatBoost   - native ``save_model(format="onnx")`` (official)
* XGBoost    - hand-rolled TreeEnsembleClassifier built from the booster JSON
               dump (``onnxmltools`` is incompatible with xgboost >= 3)
* LightGBM   - hand-rolled TreeEnsembleClassifier built from ``dump_model()``
* RandomForest / ExtraTrees / HistGB / MLP - via skl2onnx (best-effort)

TabPFN has no ONNX converter, so it is excluded; ``weights_onx`` renormalises
the blend weights over the ONNX-servable models only. Every exported model is
numerically validated against the pickle's ``predict_proba`` (max abs error
< 1e-4) on synthetic input; anything that fails is dropped from the manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import onnx
from onnx import helper, TensorProto

from .config import ARTIFACT_DIR, NEEDS_IMPUTE

log = logging.getLogger("best_model.export_onnx")

_ONNX_TRIED = ("xgb", "lgbm", "cat", "histgb", "rf", "et", "mlp")


# ── gradient-boosted tree builders ───────────────────────────────────────
def _tree_ensemble_attrs(spec: dict) -> dict:
    """Flatten xgb/lgbm JSON trees into TreeEnsembleClassifier attributes.

    A fresh pre-order numbering is assigned per tree via object identity,
    because xgb (yes/no) and lgbm (split_index vs leaf_index) use non-unique
    numeric id spaces.  Every leaf also gets a 'LEAF' row in the node arrays,
    which is the convention onnxruntime expects.
    """
    mode = spec["mode"]
    base = float(spec.get("base", 0.0))
    bt, bn, bm, bf, bv = [], [], [], [], []
    bthen, belse, bmiss_t = [], [], []
    clt, cln, cli, clw = [], [], [], []

    def children_of(node):
        if mode == "xgb":
            return [c for c in node.get("children", []) if c is not None]
        return [node["left_child"], node["right_child"]]

    def is_leaf(node):
        return "leaf" in node or "leaf_value" in node

    def leaf_value(node):
        return float(node.get("leaf", node.get("leaf_value", 0.0)))

    for tid, tree in enumerate(spec["trees"]):
        root = tree if mode == "xgb" else tree.get("tree_structure", tree)
        order: list[dict] = []

        def visit(node):
            order.append(node)
            if not is_leaf(node):
                for child in children_of(node):
                    if child is not None:
                        visit(child)

        visit(root)
        id_map = {id(n): i for i, n in enumerate(order)}
        for node in order:
            nid = id_map[id(node)]
            if is_leaf(node):
                clt.append(tid); cln.append(nid); cli.append(1)
                clw.append(leaf_value(node))
                bt.append(tid); bn.append(nid); bm.append("LEAF")
                bf.append(0); bv.append(0.0)
                bthen.append(0); belse.append(0); bmiss_t.append(0)
                continue
            if mode == "xgb":
                bm.append("BRANCH_LT")
                bf.append(int(node["split"][1:]))
                bv.append(float(node["split_condition"]))
                children = node.get("children", [])
                yes_id = int(node["yes"])
                no_id = int(node["no"])
                then_obj = next(c for c in children if int(c["nodeid"]) == yes_id)
                else_obj = next(c for c in children if int(c["nodeid"]) == no_id)
                then_id = id_map[id(then_obj)]
                else_id = id_map[id(else_obj)]
                missing_id = int(node.get("missing", node["yes"]))
                missing_left = (missing_id == yes_id)
            else:
                bm.append("BRANCH_LEQ")
                bf.append(int(node["split_feature"]))
                bv.append(float(node["threshold"]))
                then_id = id_map[id(node["left_child"])]
                else_id = id_map[id(node["right_child"])]
                missing_left = bool(node.get("default_left", True))
            bt.append(tid); bn.append(nid)
            bthen.append(then_id); belse.append(else_id)
            bmiss_t.append(int(missing_left))

    if base != 0.0:
        tid = len(spec["trees"])
        clt.append(tid); cln.append(0); cli.append(1)
        clw.append(base)
        bt.append(tid); bn.append(0); bm.append("LEAF")
        bf.append(0); bv.append(0.0)
        bthen.append(0); belse.append(0); bmiss_t.append(0)

    return {
        "nodes_treeids": np.asarray(bt, dtype=np.int64),
        "nodes_nodeids": np.asarray(bn, dtype=np.int64),
        "nodes_modes": bm,
        "nodes_featureids": np.asarray(bf, dtype=np.int64),
        "nodes_values": np.asarray(bv, dtype=np.float32),
        "nodes_truenodeids": np.asarray(bthen, dtype=np.int64),
        "nodes_falsenodeids": np.asarray(belse, dtype=np.int64),
        "nodes_missing_value_tracks_true": np.asarray(bmiss_t, dtype=np.int64),
        "class_treeids": np.asarray(clt, dtype=np.int64),
        "class_nodeids": np.asarray(cln, dtype=np.int64),
        "class_ids": np.asarray(cli, dtype=np.int64),
        "class_weights": np.asarray(clw, dtype=np.float32),
        "base_values": np.asarray([0.0], dtype=np.float32),
        "post_transform": "LOGISTIC",
        "classlabels_int64s": [0, 1],
    }


def _convert_gbt(est, name: str, n_features: int) -> onnx.ModelProto:
    """Build an ONNX TreeEnsembleClassifier from an XGB/LGBM booster."""
    import json as _json

    if name == "xgb":
        neb = est.get_booster()
        trees = [_json.loads(t) for t in neb.get_dump(dump_format="json")]
        if neb.best_iteration is not None and neb.best_iteration >= 0:
            n_keep = int(neb.best_iteration) + 1
            if n_keep < len(trees):
                log.info("xgb: best_iteration=%d -> using first %d of %d trees",
                         neb.best_iteration, n_keep, len(trees))
                trees = trees[:n_keep]
        try:
            cfg = _json.loads(neb.save_config())
            base = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))
        except Exception:
            base = float(est.get_params().get("base_score") or 0.5)
        if base <= 0 or base >= 1:
            base = 0.5
        base_margin = math.log(base / (1.0 - base))
        spec = {"trees": trees, "base": base_margin, "mode": "xgb"}
    else:
        d = est.booster_.dump_model()
        spec = {"trees": d.get("tree_info", []),
                "base": float(d.get("average_output", 0.0)),
                "mode": "lgbm"}
    if not spec["trees"]:
        raise ValueError("empty tree dump")
    attrs = _tree_ensemble_attrs(spec)
    node = helper.make_node(
        "TreeEnsembleClassifier", ["features"], ["label", "treens_out"],
        name="TreeEnsembleClassifier", domain="ai.onnx.ml", **attrs,
    )
    graph = helper.make_graph(
        [node],
        "best_model_trees",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, n_features])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
         helper.make_tensor_value_info("treens_out", TensorProto.FLOAT, [None, 2])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("ai.onnx.ml", 4),
                              helper.make_opsetid("ai.onnx", 17)])


def _convert_sklearn(est, n_features: int):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    return convert_sklearn(est,
                           initial_types=[("features", FloatTensorType([None, n_features]))],
                           target_opset=17)


def _stats_of(imp) -> dict:
    med = np.asarray(imp._imp.statistics_, dtype=np.float64)
    scale = np.asarray(imp._scaler.scale_, dtype=np.float64)
    mean = np.asarray(imp._scaler.mean_, dtype=np.float64)
    return {
        "n_features": int(med.size),
        "median": med.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


def _p1_of(outs, n_rows: int) -> np.ndarray:
    """Extract class-1 probability from an ONNX output list regardless of layout.

    Prioritises float-typed 2-col outputs (boosters), then float 1-col, and
    skips label outputs (int64 / object).
    """
    for o in outs:
        a = np.asarray(o)
        if a.dtype.kind == "O":
            try:
                return np.array([float(d[1]) for d in a], dtype=np.float64)
            except (TypeError, KeyError, IndexError):
                continue
        if a.dtype.kind not in ("f", "d", "e"):
            continue
        if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] == n_rows:
            return a[:, 1].astype(np.float64)
    for o in outs:
        a = np.asarray(o)
        if a.dtype.kind in ("O",) or a.dtype.kind not in ("f", "d", "e"):
            continue
        if a.ndim == 1 and a.shape[0] == n_rows:
            return a.astype(np.float64)
        if a.ndim == 2 and a.shape[1] == 1 and a.shape[0] == n_rows:
            return a[:, 0].astype(np.float64)
    raise ValueError("p1 output unresolved")


def _verify(sess, est, n_features: int, impute: dict | None) -> tuple[float, bool]:
    """(max |onnx - sklearn|, flip_needed).  Some exporters emit P(class=0)."""
    rng = np.random.RandomState(0)
    X = rng.randn(64, n_features).astype(np.float64)
    X[::3, :] = np.nan
    if impute:
        X = (np.nan_to_num(X, nan=np.asarray(impute["median"]))
             - np.asarray(impute["mean"])) / np.asarray(impute["scale"])
    Xf = X.astype(np.float32)
    outs = sess.run(None, {"features": Xf})
    p_onx = _p1_of(outs, X.shape[0])
    p_sk = est.predict_proba(X)[:, 1].astype(np.float64)
    err = float(np.max(np.abs(p_onx - p_sk)))
    if err > 1e-4:
        err_flip = float(np.max(np.abs((1.0 - p_onx) - p_sk)))
        if err_flip < err:
            return err_flip, True
    return err, False


def export_onnx(best: object, artifact_dir: Path, out_dir: Path) -> Path:
    """Convert every convertible base model and write ``manifest.json``."""
    import onnxruntime as ort

    out_dir.mkdir(parents=True, exist_ok=True)
    models = getattr(best, "models", {})
    manifest = {
        "mode": getattr(best, "mode", "blend"),
        "threshold": float(getattr(best, "threshold", 0.5)),
        "weights": {},
        "weights_onx": {},
        "models": {},
    }
    onx_w = 0.0
    for name, est in models.items():
        if name not in _ONNX_TRIED or not hasattr(est, "predict_proba"):
            continue
        n_feat = int(getattr(est, "n_features_in_", 0) or 0)
        if n_feat <= 0:  # CatBoost exposes n_features_in_ as 0
            try:
                n_feat = len(est.get_feature_importance())
            except Exception:
                n_feat = 0
        if n_feat <= 0:
            log.warning("drop %s: cannot resolve n_features", name)
            continue
        w = float(getattr(best, "weights", {}).get(name, 0.0))
        entry: dict = {"file": f"{name}.onnx", "n_features": n_feat, "weight": w}
        try:
            if name == "cat":
                fn_onnx = out_dir / entry["file"]
                est.save_model(str(fn_onnx), format="onnx",
                               export_parameters={"onnx_domain": "ai.onnx.ml"})
                err, flip = _verify(ort.InferenceSession(str(fn_onnx)), est, n_feat, None)
                entry["p1_is_1_minus_output"] = bool(flip)
            elif name in ("xgb", "lgbm"):
                fn_onnx = out_dir / entry["file"]
                onnx.save(_convert_gbt(est, name, n_feat), str(fn_onnx))
                err, flip = _verify(ort.InferenceSession(str(fn_onnx)), est, n_feat, None)
            else:
                impute = getattr(best, "imputer_scale", {}).get(name)
                if impute is not None and name in NEEDS_IMPUTE:
                    entry["impute"] = _stats_of(impute)
                    entry["impute_kind"] = "median+standard"
                fn_onnx = out_dir / entry["file"]
                onnx.save(_convert_sklearn(est, n_feat), str(fn_onnx))
                err, flip = _verify(ort.InferenceSession(str(fn_onnx)), est, n_feat,
                                    entry.get("impute"))
            if err > 1e-4:
                log.warning("drop %s: onnx/sklearn max err %.6g > 1e-4", name, err)
                continue
            log.info("exported %-5s n_feat=%d weight=%.4f maxerr=%.2e flip=%s",
                     name, n_feat, w, err, flip)
            manifest["models"][name] = entry
            manifest["weights"][name] = w
            onx_w += w
        except Exception as exc:
            log.warning("drop %s: %s: %s", name, type(exc).__name__, str(exc)[:180])
            continue

    for name, w in manifest["weights"].items():
        manifest["weights_onx"][name] = (w / onx_w) if onx_w > 0 else 0.0
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return mpath


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Export trained best_model to ONNX + manifest.")
    ap.add_argument("--artifacts", default=str(ARTIFACT_DIR), help="dir with best_model.pkl")
    ap.add_argument("--out", default=None, help="output dir (default <artifacts>/onnx)")
    args = ap.parse_args()

    artifact_dir = Path(args.artifacts)
    pkl = artifact_dir / "best_model.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"missing {pkl}; run best_model.train first")

    import sys
    from . import train as _train_mod

    _train_mod.BestEnsemble
    sys.modules.setdefault("best_model.train", _train_mod)
    for _mname in ("__main__", "best_model.export_onnx", "best_model.train"):
        _m = sys.modules.get(_mname)
        if _m is not None and not hasattr(_m, "BestEnsemble"):
            try:
                _m.BestEnsemble = _train_mod.BestEnsemble
            except Exception:
                pass
    best = joblib.load(pkl)
    if type(best).__module__ != "best_model.train":
        type(best).__module__ = "best_model.train"
        joblib.dump(best, pkl, compress=3)
        log.info("re-saved %s with canonical class module", pkl)

    out_dir = Path(args.out) if args.out else artifact_dir / "onnx"
    mpath = export_onnx(best, artifact_dir, out_dir)
    log.info("manifest written to %s", mpath)


if __name__ == "__main__":
    main()