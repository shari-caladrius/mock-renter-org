#!/usr/bin/env python3
"""renter_spec.py - turns a renter input JSON into the engine spec, runs the engine, verifies the result.

Used by populate_renter.py (the command you run). Standard library only; MongoDB access lives in populate_renter.py.

Input keys handled here (see README.md for the full reference):
    org, seed, snapshot
    gpu                 fleet default GPU: a gpu_catalog.json key ("H200", "GB200", "B300", ...) or an object
    gpuMetrics          fleet default ranges: temperatureC / powerW / memoryUsedPct, each [min, max] (GPUs running a
                        workload) or {"idle": [min, max], "busy": [min, max]}
    clusters            [{name, kind, nodes, alias?, sharedFs?, gpu?, gpuMetrics?, workloads?}] or {count, slurm, k8s, nodes}
                        nodes and workload counts take a number or a range [min, max] (drawn from the seed)
    workloadsPerCluster default workload counts for clusters without their own "workloads"
    workloads           {"items": [{name, cluster, kind, gpus}]} - pins exact workloads instead of counts
    workloadHealth      optional; absent = every workload Healthy
    bandPct / bandTargets, storage, utilization, trend, overrides, agentSyncOffsetS
"""

import datetime as dt
import hashlib
import json
import random
import re
import sys
import types
import uuid
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "engine"
CATALOG_FILE = HERE / "gpu_catalog.json"
MARKER_COLLECTION = "_mock_org_tool"
KINDS = ("TRAINING_JOB", "INFERENCE_SERVICE")


# ===================================================================================================================== helpers
class RequestError(Exception):
    """Problem in the request; message is shown to the user as-is."""


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj, indent=2):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
        f.write("\n")


def sha16(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def rng_for(*parts):
    return random.Random(int(hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16))


def det_uuid(*parts):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "mock-org|" + "|".join(str(p) for p in parts)))


def det_slug(*parts, n=11):
    h = int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest(), 16)
    alphabet, out = "abcdefghijklmnopqrstuvwxyz0123456789", ""
    while len(out) < n:
        h, d = divmod(h, 36)
        out += alphabet[d]
    return "s" + out[1:] if out[0].isdigit() else out


def snapshot_of(req):
    """'now' (default) -> current UTC second; otherwise an ISO string 'YYYY-MM-DDTHH:MM:SSZ'."""
    s = req.get("snapshot") or "now"
    if s == "now":
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None)
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise RequestError(f"snapshot must be 'now' or 'YYYY-MM-DDTHH:MM:SSZ', got {s!r}")


def epoch(d):
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp())


def as_range(v, where):
    """Integer or [min, max] -> [min, max]; None stays None."""
    if v is None:
        return None
    if isinstance(v, bool):
        raise RequestError(f"{where}: expected a number or [min, max], got {v!r}")
    if isinstance(v, int):
        return [v, v]
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v) and v[0] <= v[1]:
        return list(v)
    raise RequestError(f"{where}: expected an integer or an integer range [min, max], got {v!r}")


def as_frange(v, where, lo_limit, hi_limit):
    """[min, max] of numbers inside [lo_limit, hi_limit]."""
    ok = (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))
    if not ok or not (lo_limit <= v[0] <= v[1] <= hi_limit):
        raise RequestError(f"{where}: expected [min, max] with {lo_limit} <= min <= max <= {hi_limit}, got {v!r}")
    return [float(v[0]), float(v[1])]


def largest_remainder(total, weights):
    """Split an integer total proportionally to weights (all >= 0); result sums exactly to total."""
    s = float(sum(weights))
    if total == 0 or s == 0:
        return [0] * len(weights)
    raw = [total * w / s for w in weights]
    out = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: -(raw[i] - out[i]))[:total - sum(out)]:
        out[i] += 1
    return out


def check_keys(obj, allowed, where):
    if not isinstance(obj, dict):
        raise RequestError(f"{where}: must be an object")
    extra = sorted(set(obj) - set(allowed) - {"_comment"})
    if extra:
        raise RequestError(f"{where}: unknown key(s) {extra}; allowed: {sorted(allowed)}")


def resolve_org(req):
    org = dict(req.get("org") or {})
    name = org.get("orgName") or org.get("name")
    if not name:
        raise RequestError("org.orgName is required (it builds referenceDb = orgName.projectName.discover)")
    org.pop("name", None)
    org["orgName"] = name
    org.setdefault("orgId", det_uuid("org", name))
    org.setdefault("orgSlug", det_slug("slug", name))
    org.setdefault("createdBy", det_uuid("user", name))
    org.setdefault("updatedBy", org["createdBy"])
    org.setdefault("provider", {"label": "Chronoscale", "value": "chronoscale"})
    org.setdefault("region", "global")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,40}", org["orgSlug"]):
        raise RequestError(f"org.orgSlug {org['orgSlug']!r}: use 3-41 lowercase letters, digits, '-' or '_'")
    return org


# ===================================================================================================================== GPUs
GPU_KEYS = {"type", "model", "memoryMib", "gpusPerNode", "priceCentsPerHour", "idlePowerW", "maxPowerW", "cooling"}
DEFAULT_GPU = "H200"
METRICS = {"temperatureC": (0.0, 110.0), "powerW": (1.0, 3000.0), "memoryUsedPct": (0.0, 100.0)}   # hard limits
TEMP_DEFAULT = {"air": {"idle": [28.0, 38.0], "busy": [40.0, 83.0]}, "liquid": {"idle": [25.0, 34.0], "busy": [32.0, 68.0]}}
MEM_DEFAULT = {"idle": [1.5, 5.0], "busy": [45.0, 90.0]}


def gpu_catalog():
    return {k: v for k, v in load_json(CATALOG_FILE).items() if not k.startswith("_")}


def resolve_gpu(v, where, catalog):
    """"GB200" | {"type": "GB200", <overrides>} | {"model", "memoryMib", "gpusPerNode", "priceCentsPerHour", ...}."""
    if v is None:
        v = DEFAULT_GPU
    if isinstance(v, str):
        v = {"type": v}
    check_keys(v, GPU_KEYS, where)
    base = None
    if "type" in v:
        base = catalog.get(str(v["type"]).upper())
        if base is None:
            raise RequestError(f"{where}.type {v['type']!r} is not in gpu_catalog.json; known: {sorted(catalog)}")
        gtype = str(v["type"]).upper()
    else:                                     # a model string such as "NVIDIA H200 141GB HBM3e" -> its catalog entry
        gtype = next((k for k, c in catalog.items() if c["model"] == v.get("model")), None) or \
            next((k for k in sorted(catalog, key=len, reverse=True) if re.search(rf"\b{k}\b", str(v.get("model", "")), re.I)), "CUSTOM")
        base = catalog.get(gtype)
    g = dict(base or {})
    g.update({k: x for k, x in v.items() if k != "type"})
    g["type"] = gtype
    missing = [k for k in ("model", "memoryMib", "gpusPerNode", "priceCentsPerHour") if k not in g]
    if missing:
        raise RequestError(f"{where}: unknown GPU, give \"type\" (one of {sorted(catalog)}) or all of {missing}")
    for k in ("memoryMib", "gpusPerNode", "priceCentsPerHour"):
        if not isinstance(g[k], int) or isinstance(g[k], bool) or g[k] < 1:
            raise RequestError(f"{where}.{k} must be a positive integer")
    g.setdefault("cooling", "air")
    if g["cooling"] not in TEMP_DEFAULT:
        raise RequestError(f"{where}.cooling must be 'air' or 'liquid'")
    g.setdefault("idlePowerW", round(g.get("maxPowerW", 700) * 0.1))
    g.setdefault("maxPowerW", 700)
    return g


def default_ranges(gpu):
    idle_w, max_w = float(gpu["idlePowerW"]), float(gpu["maxPowerW"])
    return {"temperatureC": {k: list(x) for k, x in TEMP_DEFAULT[gpu["cooling"]].items()},
            "powerW": {"idle": [round(idle_w * 0.9), round(idle_w * 1.25)], "busy": [round(idle_w * 1.3), max_w]},
            "memoryUsedPct": {k: list(x) for k, x in MEM_DEFAULT.items()}}


def apply_ranges(ranges, block, where):
    """block = {"temperatureC": [min, max] | {"idle": [..], "busy": [..]}, ...}; [min, max] alone = GPUs in a workload."""
    if not block:
        return ranges
    check_keys(block, set(METRICS), where)
    for m, v in block.items():
        lo, hi = METRICS[m]
        if isinstance(v, list):
            v = {"busy": v}
        check_keys(v, {"idle", "busy"}, f"{where}.{m}")
        for state, r in v.items():
            ranges[m][state] = as_frange(r, f"{where}.{m}.{state}", lo, hi)
    return ranges


# ===================================================================================================================== clusters
CLUSTER_NAMES = ["helios", "atlas", "orion", "vega", "lyra", "nova", "draco", "cygnus", "perseus", "aquila", "carina", "hydra",
                 "lynx", "pegasus", "sirius", "rigel", "altair", "deneb", "polaris", "castor", "pollux", "auriga", "corvus", "lupus"]
INFER_MODELS = ["deepseek-v3-dynamo-sglang-inference", "llama-3-405b-dynamo-trtllm-inference", "qwen2-72b-dynamo-vllm-endpoint",
                "mixtral-8x7b-dynamo-service", "command-r-plus-dynamo-tgi-api", "gemma-2-27b-dynamo-ray-serve-inference",
                "deepseek-v3-dynamo-vllm-serve"]
TRAIN_MODELS = ["llama-3-70b-pretrain", "mistral-7b-instruct-sft", "llama-3-8b-continued-pretrain", "mixtral-8x22b-pretrain",
                "qwen2-7b-dpo", "gemma-2-9b-sft"]
DEFAULT_WORKLOADS = {"TRAINING_JOB": [0, 3], "INFERENCE_SERVICE": [5, 8]}
CLUSTER_KEYS = {"name", "alias", "kind", "nodes", "sharedFs", "gpu", "gpuMetrics", "workloads"}
UTIL_MODEL = {                                     # (mean, sd, min, max) utilisation % per workload kind and band
    ("INFERENCE_SERVICE", "under"): (14.0, 4.0, 1.0, 19.9),
    ("INFERENCE_SERVICE", "normal"): (45.0, 14.0, 22.0, 78.0),
    ("INFERENCE_SERVICE", "over"): (88.0, 5.0, 80.5, 99.5),
    ("TRAINING_JOB", "under"): (14.0, 4.0, 1.0, 19.9),
    ("TRAINING_JOB", "normal"): (62.0, 8.0, 40.0, 79.0),
    ("TRAINING_JOB", "over"): (91.0, 4.0, 82.0, 99.7),
}
BAND_PREF = {"TRAINING_JOB": {"under": 0.02, "normal": 0.60, "over": 0.38},
             "INFERENCE_SERVICE": {"under": 0.78, "normal": 0.14, "over": 0.08}}
# Issue scenarios for the optional workloadHealth block; value ["x", f] = healthy value times f.
SCENARIOS = {
    ("INFERENCE_SERVICE", "failing"): [
        {"xid_errors": ["Failing", 1], "ecc_uncorrectable": ["Failing", 1], "pod_restart_count": ["Failing", 1],
         "inference_failure_count": ["Failing", 0.8]}],
    ("INFERENCE_SERVICE", "degrading"): [
        {"container_termination": ["Degrading", 1], "pod_restart_count": ["Degrading", 1], "inference_failure_count": ["Degrading", 0.6]},
        {"pod_restart_count": ["Degrading", 18], "container_termination": ["Degrading", 18], "inference_failure_count": ["Degrading", 1.0]},
        {"ttft": ["Degrading", ["x", 1.8]], "queue_wait_proxy": ["Degrading", 14.0], "gpu_kv_cache_pct": ["Degrading", 96.0],
         "requests_waiting": ["Degrading", 18.0], "kv_cache_eviction_rate": ["Degrading", 4.2]},
        {"compute_vs_queue_ratio": ["Degrading", ["x", 0.59]], "tpot": ["Degrading", ["x", 1.69]]}],
    ("TRAINING_JOB", "failing"): [
        {"xid_errors": ["Failing", 1], "ecc_uncorrectable": ["Failing", 1], "ecc_correctable_rate": ["Degrading", 0.3],
         "nccl_error_count": ["Failing", 1], "tokens_per_second_per_gpu": ["Failing", 0.0],
         "checkpoint_age": ["Degrading", 58.0], "checkpoint_write_duration": ["Degrading", 212.4]}],
    ("TRAINING_JOB", "degrading"): [
        {"oom_killed_count": ["Degrading", 1], "job_requeue_count": ["Degrading", 1], "checkpoint_age": ["Degrading", 40.0]}],
}
RENTER_KEYS = {"orgType", "org", "seed", "snapshot", "gpu", "gpuMetrics", "clusters", "workloadsPerCluster", "workloads",
               "workloadHealth", "bandTargets", "bandPct", "storage", "utilization", "trend", "agentSyncOffsetS", "overrides",
               "target"}


def workload_counts(block, where):
    """{"TRAINING_JOB": 2 | [0, 3], "INFERENCE_SERVICE": ...} -> {kind: [min, max]}."""
    check_keys(block, set(KINDS), where)
    out = {}
    for k in KINDS:
        r = as_range(block.get(k, 0), f"{where}.{k}")
        if r[0] < 0:
            raise RequestError(f"{where}.{k}: counts cannot be negative")
        out[k] = r
    return out


def renter_clusters(req, seed, catalog):
    gpu_default = req.get("gpu")
    metrics_default = req.get("gpuMetrics") or {}
    wl_default = workload_counts(req.get("workloadsPerCluster") or DEFAULT_WORKLOADS, "workloadsPerCluster")
    c = req.get("clusters")
    if isinstance(c, list):
        raw = []
        for i, x in enumerate(c):
            where = f"clusters[{i}]"
            check_keys(x, CLUSTER_KEYS, where)
            if not isinstance(x.get("name"), str):
                raise RequestError(f"{where}.name is required")
            if x.get("kind") not in ("slurm", "k8s"):
                raise RequestError(f"{where}.kind must be 'slurm' or 'k8s'")
            nodes = as_range(x.get("nodes"), f"{where}.nodes")
            if not nodes or nodes[0] < 1:
                raise RequestError(f"{where}.nodes must be a positive integer or a range [min, max]")
            raw.append({**x, "nodes": nodes})
    elif isinstance(c, dict):
        check_keys(c, {"count", "slurm", "k8s", "nodes"}, "clusters")
        n_s, n_k, total = c.get("slurm"), c.get("k8s"), c.get("count")
        if total is None:
            total = (n_s or 0) + (n_k or 0)
        if n_s is None and n_k is None:
            n_s = max(1, total // 3)
        n_s = n_s if n_s is not None else total - n_k
        n_k = n_k if n_k is not None else total - n_s
        if n_s + n_k != total or total < 1:
            raise RequestError("clusters: count must equal slurm + k8s and be at least 1")
        nodes = as_range(c.get("nodes", [7, 12]), "clusters.nodes")
        names = CLUSTER_NAMES[:total] if total <= len(CLUSTER_NAMES) else [f"cluster-{i + 1:02d}" for i in range(total)]
        raw = [{"name": nm, "kind": k, "nodes": nodes} for nm, k in zip(names, ["slurm"] * n_s + ["k8s"] * n_k)]
    else:
        raise RequestError("clusters: give a list of clusters or {count, slurm, k8s, nodes}")
    out = []
    for i, x in enumerate(raw):
        where, name = f"clusters[{i}]", x["name"]
        gpu = resolve_gpu(x.get("gpu", gpu_default), f"{where}.gpu" if "gpu" in x else "gpu", catalog)
        ranges = apply_ranges(default_ranges(gpu), metrics_default, "gpuMetrics")
        ranges = apply_ranges(ranges, x.get("gpuMetrics"), f"{where}.gpuMetrics")
        wl_range = workload_counts(x["workloads"], f"{where}.workloads") if "workloads" in x else wl_default
        r = rng_for(seed, "renter-cluster", name)
        out.append({"name": name, "alias": x.get("alias") or name.title(), "kind": x["kind"],
                    "nodes": r.randint(*x["nodes"]), "nodesAsked": x["nodes"], "sharedFs": x.get("sharedFs"),
                    "gpu": gpu, "ranges": ranges, "workloadsAsked": wl_range,
                    "workloadCounts": {k: rng_for(seed, "renter-wl-count", name, k).randint(*wl_range[k]) for k in KINDS}})
    return out


# ===================================================================================================================== workloads
def fit_bands(sizes, kinds, targets):
    """Integer matrix workloads x {under, normal, over}: rows sum to sizes, columns sum to targets, close to BAND_PREF."""
    bands = ("under", "normal", "over")
    m = [[sizes[i] * BAND_PREF[kinds[i]][b] for b in bands] for i in range(len(sizes))]
    for _ in range(200):                                                        # iterative proportional fitting
        for j, b in enumerate(bands):
            col = sum(row[j] for row in m)
            f = targets[b] / col if col else 0
            for row in m:
                row[j] *= f
        for i, row in enumerate(m):
            s = sum(row)
            f = sizes[i] / s if s else 0
            for j in range(3):
                row[j] *= f
    out = [largest_remainder(sizes[i], m[i]) for i in range(len(sizes))]
    for _ in range(100000):                                                     # make the columns exact
        cur = {b: sum(row[j] for row in out) for j, b in enumerate(bands)}
        hi = [j for j, b in enumerate(bands) if cur[b] > targets[b]]
        lo = [j for j, b in enumerate(bands) if cur[b] < targets[b]]
        if not hi:
            break
        a, z = hi[0], lo[0]
        i = max((i for i in range(len(out)) if out[i][a] > 0),
                key=lambda i: BAND_PREF[kinds[i]][bands[z]] - BAND_PREF[kinds[i]][bands[a]])
        out[i][a] -= 1
        out[i][z] += 1
    return [dict(zip(bands, row)) for row in out]


def renter_workloads(req, clusters, band_targets, seed):
    wreq = req.get("workloads") or {}
    if set(wreq) & set(KINDS):
        raise RequestError("workload counts are set per cluster now: put \"workloads\": {\"TRAINING_JOB\": [0, 3], "
                           "\"INFERENCE_SERVICE\": [5, 8]} inside each cluster, or \"workloadsPerCluster\" for all clusters. "
                           "Top-level \"workloads\" only takes {\"items\": [...]} (exact workloads)")
    check_keys(wreq, {"items"}, "workloads")
    cl = {c["name"]: c for c in clusters}
    if wreq.get("items"):
        items = []
        for i, w in enumerate(wreq["items"]):
            check_keys(w, {"name", "cluster", "kind", "gpus", "buckets"}, f"workloads.items[{i}]")
            if w.get("cluster") not in cl:
                raise RequestError(f"workloads.items[{i}]: unknown cluster {w.get('cluster')!r}")
            if w.get("kind") not in KINDS:
                raise RequestError(f"workloads.items[{i}]: kind must be TRAINING_JOB or INFERENCE_SERVICE")
            if not isinstance(w.get("gpus"), int) or w["gpus"] < 1:
                raise RequestError(f"workloads.items[{i}].gpus must be a positive integer")
            items.append(dict(w))
        if len({w["name"] for w in items}) != len(items):
            raise RequestError("workloads.items: workload names must be unique")
        if not all("buckets" in w for w in items):
            for w, b in zip(items, fit_bands([w["gpus"] for w in items], [w["kind"] for w in items], band_targets)):
                w["buckets"] = b
        return items
    gpus = {c["name"]: c["nodes"] * c["gpu"]["gpusPerNode"] for c in clusters}
    per = {c["name"]: ["TRAINING_JOB"] * c["workloadCounts"]["TRAINING_JOB"] +
           ["INFERENCE_SERVICE"] * c["workloadCounts"]["INFERENCE_SERVICE"] for c in clusters}
    if not any(per.values()):
        raise RequestError("no workloads: every cluster drew 0 workloads; raise the counts in clusters[].workloads / workloadsPerCluster")
    for n, kinds in per.items():
        if len(kinds) > gpus[n]:
            raise RequestError(f"cluster {n}: {len(kinds)} workloads but only {gpus[n]} GPUs; lower its workload counts")
    # idle per cluster: clusters without workloads are fully idle, the rest share the remaining idle target
    idle_total = band_targets["idle"]
    forced = sum(gpus[n] for n in per if not per[n])
    if forced > idle_total:
        raise RequestError(f"{forced} GPUs sit in clusters with no workload but the idle target is {idle_total}; "
                           f"add workloads or raise bandPct.idle")
    active = [c["name"] for c in clusters if per[c["name"]]]
    cap = {n: min(gpus[n] - len(per[n]), min(5, cl[n]["gpu"]["gpusPerNode"]) * cl[n]["nodes"]) for n in active}
    idle = dict(zip(active, largest_remainder(idle_total - forced, [gpus[n] for n in active])))
    for _ in range(len(active) + 2):                                           # respect per-cluster caps
        over = sum(max(0, idle[n] - cap[n]) for n in active)
        if not over:
            break
        for n in active:
            idle[n] = min(idle[n], cap[n])
        room = [n for n in active if idle[n] < cap[n]]
        if not room:
            raise RequestError("idle target too high for the requested workloads; lower bandPct.idle or add workloads")
        for n, k in zip(room, largest_remainder(over, [cap[n] - idle[n] for n in room])):
            idle[n] += k
    if sum(max(0, idle[n] - cap[n]) for n in active):
        raise RequestError("idle target too high for the requested workloads; lower bandPct.idle or add workloads")
    items, used = [], Counter()
    weight = {"TRAINING_JOB": 3.0, "INFERENCE_SERVICE": 1.0}
    for c in clusters:
        n, kinds = c["name"], per[c["name"]]
        if not kinds:
            continue
        busy = gpus[n] - idle[n]
        r = rng_for(seed, "renter-wl", n)
        sizes = [1 + x for x in largest_remainder(busy - len(kinds), [weight[k] * r.uniform(0.5, 1.5) for k in kinds])]
        for k, g in zip(kinds, sizes):
            pool = TRAIN_MODELS if k == "TRAINING_JOB" else INFER_MODELS
            base = pool[used[(n, k)] % len(pool)]
            suffix = "" if used[(n, k)] < len(pool) else f"-{used[(n, k)] // len(pool) + 1}"
            used[(n, k)] += 1
            items.append({"name": f"{base}-{n}{suffix}", "cluster": n, "kind": k, "gpus": g})
    order = {"TRAINING_JOB": 0, "INFERENCE_SERVICE": 1}
    items.sort(key=lambda w: (order[w["kind"]], [c["name"] for c in clusters].index(w["cluster"])))
    for w, b in zip(items, fit_bands([w["gpus"] for w in items], [w["kind"] for w in items], band_targets)):
        w["buckets"] = b
    return items


def tuple_values(metrics):
    return {k: (h, tuple(v) if isinstance(v, list) else v) for k, (h, v) in metrics.items()}


def renter_health(req, workloads, seed):
    """Absent workloadHealth = every workload Healthy (the data pipeline computes the real health)."""
    hreq = req.get("workloadHealth") or {}
    check_keys(hreq, {"failing", "degrading", "items"}, "workloadHealth")
    names = {w["name"]: w for w in workloads}
    if hreq.get("items"):
        out = {}
        for n, x in hreq["items"].items():
            if n not in names:
                raise RequestError(f"workloadHealth.items: unknown workload {n!r}")
            if x.get("state") not in ("failing", "degrading"):
                raise RequestError(f"workloadHealth.items.{n}.state must be 'failing' or 'degrading'")
            out[n] = {"state": x["state"], "metrics": tuple_values(x["metrics"])}
        return out
    nf, nd = int(hreq.get("failing", 0)), int(hreq.get("degrading", 0))
    if nf + nd > len(workloads):
        raise RequestError(f"workloadHealth asks for {nf + nd} unhealthy workloads but only {len(workloads)} exist")
    order = list(workloads)
    rng_for(seed, "renter-health").shuffle(order)
    out, used = {}, Counter()
    for state, n in (("failing", nf), ("degrading", nd)):
        for w in order[:n]:
            lib = SCENARIOS[(w["kind"], state)]
            out[w["name"]] = {"state": state, "metrics": tuple_values(lib[used[(w["kind"], state)] % len(lib)])}
            used[(w["kind"], state)] += 1
        order = order[n:]
    return out


# ===================================================================================================================== spec
def renter_spec(req, org, snap):
    check_keys(req, RENTER_KEYS, "request")
    seed = int(req.get("seed", 20261022))
    catalog = gpu_catalog()
    clusters = renter_clusters(req, seed, catalog)
    names = [c["name"] for c in clusters]
    if len(set(names)) != len(names):
        raise RequestError(f"duplicate cluster names: {names}")
    forbidden = ("slurm", "k8s", "kubernetes") + tuple(k.lower() for k in catalog)
    for n in names:
        if not re.fullmatch(r"[a-z][a-z0-9-]{1,30}", n) or any(t in n for t in forbidden):
            raise RequestError(f"cluster name {n!r}: lowercase word without GPU-model or orchestrator tokens {forbidden}")
    total = sum(c["nodes"] * c["gpu"]["gpusPerNode"] for c in clusters)
    if req.get("bandTargets"):
        bt = dict(req["bandTargets"])
        if sorted(bt) != ["idle", "normal", "over", "under"] or sum(bt.values()) != total:
            raise RequestError(f"bandTargets must give idle/under/normal/over summing to {total} GPUs")
    else:
        pct = {"idle": 20, "under": 50, "normal": 20, "over": 10}
        bp = req.get("bandPct") or {}
        check_keys(bp, set(pct), "bandPct")
        pct.update(bp)
        keys = ["idle", "under", "normal", "over"]
        bt = dict(zip(keys, largest_remainder(total, [pct[k] for k in keys])))
    workloads = renter_workloads(req, clusters, bt, seed)
    per_cluster = Counter()
    for w in workloads:
        if sum(w["buckets"].values()) != w["gpus"]:
            raise RequestError(f"{w['name']}: buckets must sum to its {w['gpus']} GPUs")
        per_cluster[w["cluster"]] += w["gpus"]
    for c in clusters:
        cap = c["nodes"] * c["gpu"]["gpusPerNode"]
        if per_cluster[c["name"]] > cap:
            raise RequestError(f"{c['name']}: workloads use {per_cluster[c['name']]} GPUs, the cluster has {cap}")
        c["workloadCounts"] = {k: sum(1 for w in workloads if w["cluster"] == c["name"] and w["kind"] == k) for k in KINDS}
    if total - sum(per_cluster.values()) != bt["idle"]:
        raise RequestError(f"idle GPUs ({total - sum(per_cluster.values())}) must equal bandTargets.idle ({bt['idle']})")
    health = renter_health(req, workloads, seed)
    st = {"k8sScratchGibChoices": [2048, 4096], "fleetTargetPct": 46.0, "fleetRangePct": [42.0, 50.0]}
    st.update(req.get("storage") or {})
    trend = {"window_days": 30, "history_fleet_util_pct_range": [15.0, 40.0], "min_day_change_pct": 5.0, "week_days": 7,
             "week_fleet_util_pct_range": [15.0, 50.0], "week_min_day_change_pct": 10.0, "idle_daily_jitter_pct": 0.35}
    trend.update(req.get("trend") or {})
    overrides = req.get("overrides") or {}
    overrides.setdefault("gpu", {})
    overrides.setdefault("filesystem", {})
    ns = types.SimpleNamespace(
        SEED=seed, SNAPSHOT_UTC=snap.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ORG={"orgId": org["orgId"], "orgName": org["orgName"], "createdBy": org["createdBy"]},
        REGION=org["region"], PROVIDER=org["provider"], RACK="r01",
        AGENT_SYNC_EPOCH_S=epoch(snap) + int(req.get("agentSyncOffsetS", 298)),
        NODE_NAME="{cluster}-node-{rack}-idx{n:02d}", GPU_NAME="{node}-gpu-{k}", SCRATCH_NAME="{node}-scratch",
        SHARED_NAME="{cluster}-shared",
        CLUSTERS=[{"name": c["name"], "alias": c["alias"], "kind": c["kind"], "nodes": c["nodes"], "gpu": c["gpu"],
                   "ranges": c["ranges"]} for c in clusters],
        STORAGE={"k8s_scratch_gib_choices": st["k8sScratchGibChoices"],
                 "slurm_shared": {c["name"]: c.get("sharedFs") or {"fsType": "lustre", "capacityGib": c["nodes"] * 51200}
                                  for c in clusters if c["kind"] == "slurm"},
                 "fleet_util_pct_range": tuple(st["fleetRangePct"]), "fleet_util_pct_target": st["fleetTargetPct"]},
        UTIL_MODEL=UTIL_MODEL, BUCKET_TARGETS=bt,
        WORKLOADS=[{"name": w["name"], "cluster": w["cluster"], "kind": w["kind"], "gpus": w["gpus"], "buckets": w["buckets"],
                    "health": health[w["name"]]["state"] if w["name"] in health else "green"} for w in workloads],
        HEALTH_ISSUES=health, OVERRIDES=overrides, TREND=trend)
    expect = {"total_gpus": total,
              "clusters": [{"name": c["name"], "kind": c["kind"], "nodes": c["nodes"], "nodesAsked": c["nodesAsked"],
                            "gpu": c["gpu"], "ranges": c["ranges"], "workloads": c["workloadCounts"],
                            "workloadsAsked": None if (req.get("workloads") or {}).get("items") else c["workloadsAsked"]}
                           for c in clusters],
              "workloads": {w["name"]: (w["cluster"], w["kind"], w["gpus"]) for w in workloads}, "bands": bt,
              "health": {n: h["state"] for n, h in health.items()},
              "overridden": {n: sorted(v) for n, v in overrides["gpu"].items()},
              "fleetRangePct": (req.get("utilization") or {}).get("fleetRangePct"), "storageRangePct": st["fleetRangePct"]}
    return ns, expect


# ===================================================================================================================== build + verify
def prepare_dir(out_root, slug):
    work = Path(out_root) / slug
    if work.exists():                                   # generated output only; rebuilt every run
        for p in sorted(work.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
    work.mkdir(parents=True, exist_ok=True)
    return work


def build_renter(req, out_root):
    org = resolve_org(req)
    snap = snapshot_of(req)
    ns, expect = renter_spec(req, org, snap)
    work = prepare_dir(out_root, org["orgSlug"])
    save_json(work / "request.json", req)
    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))
    import renter_engine                                                        # noqa: E402  (engine folder on sys.path)
    gen = work / "generated"
    renter_engine.generate(ns, gen)
    save_json(work / "expect.json", expect)
    rep = verify_renter(gen, expect)
    (work / "validation_report.txt").write_text("\n".join(rep["failures"] or ["all checks passed"]) + f"\n{rep['passed']} checks passed\n")
    if rep["failures"]:
        for f in rep["failures"][:30]:
            print("  FAIL " + f)
        raise RequestError(f"renter verification failed ({len(rep['failures'])} problems); see {work / 'validation_report.txt'}")
    print(f"  validation: 0 errors ({rep['passed']} checks)")
    save_json(work / "run.json", {
        "tool": "populate_renter.py", "orgType": "RENTER", "org": org, "seed": ns.SEED,
        "snapshot": snap.strftime("%Y-%m-%dT%H:%M:%SZ"), "requestSha": sha16(json.dumps(req, sort_keys=True)),
        "engineSha": sha16(*(p.read_bytes() for p in sorted(ENGINE.glob("*.py")))),
        "builtAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    print_summary(work)
    return work


def in_range(v, r, tol=0.05):
    return r[0] - tol <= v <= r[1] + tol


def verify_renter(gen, expect):
    """Every expectation comes from the resolved request: counts, names, GPU types, metric ranges, bands, cost, storage."""
    fails, passed = [], [0]

    def check(cond, msg):
        if cond:
            passed[0] += 1
        else:
            fails.append(msg)

    L = lambda n: load_json(gen / n)                                             # noqa: E731
    ref = lambda n: load_json(ENGINE / "reference" / n)                           # noqa: E731
    ov, wh, inv = L("overview.json")["data"], L("workload-health.json")["data"], L("inventory.json")
    pc, res, mongo, na = L("project_configs.json"), L("resources.json")["data"], L("resources-mongo.json"), L("nodeagent-health.json")["data"]
    rc = L("rate-card.json")["data"]
    cfg = L("config.json")["data"]
    ecl = {c["name"]: c for c in expect["clusters"]}
    nclu = len(ecl)
    total = expect["total_gpus"]

    # identity
    check(len(pc) == len(res) == len(mongo) == len(na) == len(rc) == nclu, f"need {nclu} projects/resources/mongo docs/agents/rate cards")
    for p, r, m, n in zip(pc, res, mongo, na):
        tag = p["projectName"]
        check(r["projectId"] == p["_id"]["$oid"] == m["projectId"]["$oid"], f"{tag}: projectId chain")
        check(r["name"] == m["name"] == p["projectName"] == n["resourceConfigName"], f"{tag}: names")
        check(r["alias"] == m["alias"] == p["projectAlias"], f"{tag}: aliases")
        check(r["resourceId"] == m["resourceId"] and r["toolKey"] == m["toolKey"], f"{tag}: resourceId/toolKey")
        check(n["delegateKey"] == r["delegateKey"], f"{tag}: delegate link")
        check(p["referenceDb"] == f"{p['orgName']}.{tag}.discover", f"{tag}: referenceDb")
        c, pr, up = (int(p[k]["$numberLong"]) for k in ("createdAt", "commandCenterPreparedAt", "updatedAt"))
        check(c < pr < up, f"{tag}: timestamp order")
    for fld in ("resourceId", "toolKey", "delegateKey"):
        check(len({r[fld] for r in res}) == nclu, f"{fld} unique")
    # topology + GPU types
    check([c["name"] for c in ov["clusters"]] == [c["name"] for c in expect["clusters"]], "overview cluster order")
    gpus = ov["gpuDevices"]
    check(len(gpus) == total == ov["summary"]["totalGpuDevices"] == inv["totals"]["gpus"], f"{total} GPUs")
    check(len({g["name"] for g in gpus}) == len(gpus), "GPU names unique")
    nodes = defaultdict(list)
    for g in gpus:
        nodes[g["parent"]].append(g)
        e = ecl[g["cluster"]]["gpu"]
        check(re.fullmatch(rf"{re.escape(g['cluster'])}-node-r01-idx\d\d-gpu-\d+", g["name"]) is not None, f"{g['name']}: pattern")
        check(g["model"] == e["model"] and g["memoryTotalMib"] == e["memoryMib"], f"{g['name']}: model / memory of {e['type']}")
    check(len(nodes) == sum(c["nodes"] for c in expect["clusters"]) == ov["summary"]["totalInstances"], "node count")
    for v in nodes.values():
        gpn = ecl[v[0]["cluster"]]["gpu"]["gpusPerNode"]
        check(sorted(x["gpuIndex"] for x in v) == list(range(gpn)), f"{v[0]['parent']}: {gpn} GPUs per node")
    for c in ov["clusters"]:
        e = ecl[c["name"]]
        check(c["instanceCount"] == e["nodes"] and c["totalGpuDevices"] == e["nodes"] * e["gpu"]["gpusPerNode"], f"{c['name']}: node/GPU counts")
        check(c["infra"] == {"k8s": "K8s", "slurm": "Slurm"}[e["kind"]], f"{c['name']}: infra")
        if e["nodesAsked"]:
            check(in_range(e["nodes"], e["nodesAsked"], 0), f"{c['name']}: {e['nodes']} nodes inside {e['nodesAsked']}")
    # metric ranges (idle GPUs: idle range, GPUs in a workload: busy range; overridden values are taken as given)
    for g in gpus:
        rg, ovr = ecl[g["cluster"]]["ranges"], set(expect["overridden"].get(g["name"], []))
        state = "idle" if g["utilizationPct"] == 0 else "busy"
        vals = {"temperatureC": ("temperatureC", g["temperatureC"]), "powerW": ("powerDrawW", g["powerDrawW"]),
                "memoryUsedPct": ("memoryUsedMib", round(g["memoryUsedMib"] / g["memoryTotalMib"] * 100, 2))}
        for m, (src, v) in vals.items():
            if src not in ovr:
                check(in_range(v, rg[m][state], 0.1), f"{g['name']}: {m} {v} outside {state} range {rg[m][state]}")
    # efficiency / placement
    eff = ov["gpuEfficiency"]
    lists = {b: eff[k] for b, k in (("idle", "idle"), ("under", "underUtilized"), ("normal", "normal"), ("over", "overUtilized"))}
    check(sorted(x["name"] for v in lists.values() for x in v) == sorted(g["name"] for g in gpus), "efficiency lists partition all GPUs once")
    for b in lists:
        check(len(lists[b]) == eff[f"{b}Total"] == expect["bands"][b], f"band {b}: {len(lists[b])} == target {expect['bands'][b]}")
    e = cfg["gpuEfficiency"]
    cls = lambda u: "idle" if u <= e["idleThreshold"] else "under" if u < e["underUtilizedMax"] else "normal" if u < e["overUtilizedMin"] else "over"  # noqa: E731
    gmap = {g["name"]: g for g in gpus}
    check(all(cls(gmap[x["name"]]["utilizationPct"]) == b for b, v in lists.items() for x in v), "band membership follows config thresholds")
    wl_entries = ov["workload"]["INFERENCE_SERVICE"] + ov["workload"]["TRAINING_JOB"]
    members = Counter(d["name"] for w in wl_entries for d in w["gpu_devices"])
    check(all(v == 1 for v in members.values()), "no GPU in two workloads")
    check(set(members) == {g["name"] for g in gpus if g["utilizationPct"] > 0}, "workload GPUs == exactly the non-idle GPUs")
    check({w["name"] for w in wl_entries} == set(expect["workloads"]), "workload set equals the request")
    for w in wl_entries:
        cl, kind, n = expect["workloads"][w["name"]]
        check(w["cluster"] == cl and w["subcategory"] == kind and len(w["gpu_devices"]) == n, f"{w['name']}: cluster/kind/size")
        check(all(gmap[d["name"]]["cluster"] == cl for d in w["gpu_devices"]), f"{w['name']}: GPUs in its cluster")
    for name, c in ecl.items():
        got = Counter(w["subcategory"] for w in wl_entries if w["cluster"] == name)
        for k in KINDS:
            check(got.get(k, 0) == c["workloads"][k], f"{name}: {got.get(k, 0)} {k}, expected {c['workloads'][k]}")
            if c["workloadsAsked"]:
                check(in_range(c["workloads"][k], c["workloadsAsked"][k], 0), f"{name}: {k} count inside {c['workloadsAsked'][k]}")
    avg = round(sum(g["utilizationPct"] for g in gpus) / total, 1)
    check(abs(ov["summary"]["avgUtilizationPct"] - avg) < 0.06 and ov["summary"]["avgUtilizationPct"] == eff["avgFleetUtilPct"]
          == ov["costBreakdown"]["avgFleetUtil"], "fleet utilisation equal everywhere")
    if expect.get("fleetRangePct"):
        lo, hi = expect["fleetRangePct"]
        check(lo <= avg <= hi, f"fleet utilisation {avg} inside {lo}-{hi}")
    # storage
    st = ov["storage"]
    for f in st["filesystems"]:
        check(abs(f["usedGib"] + f["freeGib"] - f["capacityGib"]) < 0.06 and 0 < f["usedGib"] < f["capacityGib"], f"{f['name']}: sizes")
    check(sum(f["capacityGib"] for f in st["filesystems"]) == st["totalCapacityGib"], "storage capacity total")
    check(abs(round(st["totalUsedGib"] / st["totalCapacityGib"] * 100, 1) - st["utilPct"]) < 0.06, "storage util %")
    lo, hi = expect["storageRangePct"]
    check(lo <= st["utilPct"] <= hi, f"storage util {st['utilPct']} inside {lo}-{hi}")
    # cost: every GPU at its type's price; rate card per node = price x GPUs per node
    cb = ov["costBreakdown"]
    want_cost = sum(c["nodes"] * c["gpu"]["gpusPerNode"] * c["gpu"]["priceCentsPerHour"] for c in expect["clusters"]) / 100
    check(abs(cb["fleetCostPerHour"] - want_cost) < 0.005, "fleet cost = sum of GPUs x price")
    rate_sum = sum(i["priceCentsPerHour"] for d in rc for i in d["instances"]) / 100
    check(abs(rate_sum - cb["fleetCostPerHour"]) < 0.005, "rate card sums to fleet cost")
    check(abs(sum(b["costPerHour"] for b in cb["byInstance"]) - cb["fleetCostPerHour"]) < 0.005, "cost by GPU type sums to fleet cost")
    # workload health
    allw = wh["inference"]["workloads"] + wh["training"]["workloads"]
    for blk in (wh["inference"], wh["training"]):
        check(blk["total"] == len(blk["workloads"]) == blk["healthy"] + blk["degrading"] + blk["failing"], "health counters add up")
    rank = {"Healthy": 0, "Degrading": 1, "Failing": 2}
    want = {"green": "Healthy", "degrading": "Degrading", "failing": "Failing"}
    for w in allw:
        check(w["overallHealth"] == max((c["health"] for c in w["categories"]), key=rank.get), f"{w['workloadName']}: overall = worst category")
        check(w["overallHealth"] == want[expect["health"].get(w["workloadName"], "green")], f"{w['workloadName']}: requested state")
    check({w["workloadName"] for w in allw} == set(expect["workloads"]), "health covers every workload")
    # trend
    t30 = L("gpu-efficiency-trend-30d.json")["data"]["days"]
    for x in t30:
        check(x["idleCount"] + x["underCount"] + x["normalCount"] + x["overCount"] == x["totalDevices"] == total, f"{x['date']}: counts sum")
    for n in (1, 7, 15):
        check(L(f"gpu-efficiency-trend-{n}d.json")["data"]["days"] == t30[-n:], f"{n}d window == tail of 30d")
    check(t30[-1]["avgFleetUtil"] == avg and t30[-1]["idleCount"] == expect["bands"]["idle"], "last trend day == today's snapshot")
    # reference key parity
    check(set(ov) == set(ref("overview.json")["data"]), "overview top-level keys equal the reference")
    check(set(pc[0]) == set(ref("project_configs.json")[0]), "project_configs keys equal the reference")
    check(set(wh) == set(ref("workload-health.json")["data"]), "workload-health keys equal the reference")
    return {"failures": fails, "passed": passed[0]}


def summary_lines(work):
    work = Path(work)
    info = load_json(work / "run.json")
    expect = load_json(work / "expect.json")
    ov = load_json(work / "generated" / "overview.json")["data"]
    wh = load_json(work / "generated" / "workload-health.json")["data"]
    s, e, st, cb = ov["summary"], ov["gpuEfficiency"], ov["storage"], ov["costBreakdown"]
    kinds = Counter(c["infra"] for c in ov["clusters"])
    n = e["total"]
    tot = wh["training"]["total"] + wh["inference"]["total"]
    fl = wh["training"]["failing"] + wh["inference"]["failing"]
    dg = wh["training"]["degrading"] + wh["inference"]["degrading"]
    L = [f"# {info['org']['orgName']} (RENTER, slug {info['org']['orgSlug']}, snapshot {info['snapshot']})", "",
         f"GPU clusters **{s['totalClusters']}** ({kinds.get('Slurm', 0)} SLURM, {kinds.get('K8s', 0)} K8s, {s['totalGpuDevices']} GPUs, "
         f"{s['totalInstances']} instances)  ",
         f"GPU fleet utilisation **{s['avgUtilizationPct']}%**, {n - e['idleTotal']} of {n} GPUs utilised; "
         f"avg {s['avgTemperatureC']} °C, {s['avgPowerDrawW']} W, memory {s['avgMemoryUsedPct']}%  ",
         f"Storage **{st['utilPct']}%** ({st['totalUsedGib'] / 1024:.1f} / {st['totalCapacityGib'] / 1024:.0f} TiB)  ",
         f"Fleet cost **${cb['fleetCostPerHour']:,.2f}/hr**, ${cb['fleetCostPerHour'] * 24:,.2f}/day  ",
         f"GPU efficiency: idle {e['idleTotal']} ({e['idleTotal'] / n:.0%}), under {e['underTotal']} ({e['underTotal'] / n:.0%}), "
         f"normal {e['normalTotal']} ({e['normalTotal'] / n:.0%}), over {e['overTotal']} ({e['overTotal'] / n:.0%})  ",
         f"Workload health ({tot}): {tot - fl - dg} healthy, {dg} warning, {fl} critical", "",
         "| Cluster | Infra | GPU | Instances | GPUs | Training | Inference | Avg util % | Temp °C (busy) | Power W (busy) |",
         "|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    ecl = {c["name"]: c for c in expect["clusters"]}
    for c in ov["clusters"]:
        x = ecl[c["name"]]
        rg = x["ranges"]
        L.append(f"| {c['name']} | {c['infra']} | {x['gpu']['type']} | {c['instanceCount']} | {c['totalGpuDevices']} | "
                 f"{x['workloads']['TRAINING_JOB']} | {x['workloads']['INFERENCE_SERVICE']} | {c['avgUtilizationPct']} | "
                 f"{rg['temperatureC']['busy'][0]:g}-{rg['temperatureC']['busy'][1]:g} | {rg['powerW']['busy'][0]:g}-{rg['powerW']['busy'][1]:g} |")
    return L


def print_summary(work):
    lines = summary_lines(work)
    (Path(work) / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n  output: {work}")


# ===================================================================================================================== MongoDB helpers
def encode_uri(uri):
    m = re.match(r"^(mongodb(?:\+srv)?://)([^/?]*)(.*)$", uri.strip())
    if not m or "@" not in m[2]:
        return uri.strip()
    scheme, auth, rest = m.groups()
    creds, hosts = auth.rsplit("@", 1)
    from urllib.parse import quote_plus, unquote_plus
    q = lambda s: quote_plus(unquote_plus(s))                                    # noqa: E731
    user, _, pw = creds.partition(":")
    return f"{scheme}{q(user)}{':' + q(pw) if ':' in creds else ''}@{hosts}{rest}"


def mongo_client(uri):
    if uri.startswith("mongomock://"):
        import mongomock
        return mongomock.MongoClient()
    import pymongo
    c = pymongo.MongoClient(encode_uri(uri), serverSelectionTimeoutMS=10000)
    c.admin.command("ping")
    return c


def host_of(uri):
    m = re.match(r"^mongodb(?:\+srv)?://(?:[^@/]*@)?([^/?]*)", uri)
    return m.group(1) if m else uri
