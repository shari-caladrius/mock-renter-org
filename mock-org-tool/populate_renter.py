#!/usr/bin/env python3
"""populate_renter.py - build ONE mock RENTER (customer) org from a JSON file and load it into that org's MongoDB database.

Files: renter_spec.py (input -> spec, verification), engine/ (data generator), gpu_catalog.json (GPU types), requests/.

    python populate_renter.py --template > my-renter.json      # starting point
    python populate_renter.py my-renter.json --dry-run         # build + validate, JSON copies in orgs/renter/<slug>/mongo/
    MONGO_URI='mongodb+srv://user:pass@host/' \
    python populate_renter.py my-renter.json                   # same + upload (asks you to type the database name)
    python populate_renter.py my-renter.json --yes             # no question (scripts / CI)
    python populate_renter.py my-renter.json                   # with a .env next to this script: starts the kubectl port
                                                               # forward, reads user/password via kubectl, uploads, stops it
    python populate_renter.py my-renter.json --export         # no connection from here: import bundle (mongosh script +
                                                               # JSON per collection) in orgs/renter/<slug>/import/

Input JSON (every key except "org" is optional; full reference in README.md, examples in requests/):
    org          {"orgName", "orgId", "orgSlug", "region", "createdBy"}  - the REAL orgId of the new org: every document
                 carries it and the UI only shows documents of the logged-in org
    mongo        {"database": "<db of the new org>", "uri": "..."}  - prefer MONGO_URI env or --uri over a uri in the file
    seed         same seed + inputs = same data (ranges below are drawn from it)
    snapshot     "now" (trends end today) or "YYYY-MM-DDTHH:MM:SSZ"
    gpu          fleet default GPU type from gpu_catalog.json: "H200", "B200", "B300", "GB200", "GB300", "H100", ...
    gpuMetrics   fleet default ranges {"temperatureC" | "powerW" | "memoryUsedPct": [min, max] or {"idle": [..], "busy": [..]}}
    clusters     [{"name", "kind": "slurm"|"k8s", "nodes": 8 | [6, 10], "gpu"?, "gpuMetrics"?, "alias"?, "sharedFs"?,
                   "workloads"?: {"TRAINING_JOB": [0, 3], "INFERENCE_SERVICE": [5, 8]}}]
                 or {"count": 6, "slurm": 2, "k8s": 4, "nodes": [7, 12]}
    workloadsPerCluster  workload counts for clusters without their own (default TRAINING_JOB [0, 3], INFERENCE_SERVICE [5, 8])
    workloads    {"items": [{name, cluster, kind, gpus}]} - exact workloads instead of counts
    workloadHealth  optional; absent = every workload Healthy
    bandPct / bandTargets, storage, utilization, trend, overrides
    collections  {kind: "collection name" | null}  - rename or switch off a collection (kinds below)

Collections written (kind -> default name):
    entity model (upserted by _id; other documents in the collection are never touched)
        entities          caladrius_neo_cloud_entities                      cluster / node / GPU / workload / filesystem
        gpu_metrics       caladrius_neo_cloud_gpu_metrics                   one per GPU
        workload_health   caladrius_neo_cloud_workload_health_snapshots     one per workload
        rate_cards        caladrius_neo_cloud_rate_cards                    one per cluster
        resources         self_managed_resources                            one per cluster
        project_configs   project_configs                                   one per cluster (project of the resource)
        monitored_entities    caladrius_monitored_entities                  one per cluster: agent health (healthy)
        delegate_configs      delegate_configs                              one per cluster: delegate, install command, CONNECTED
        install_templates     delegate_install_templates                    one per cluster type, per-cluster values as variables
        agent_health_configs  gpu_exporter_agent_health_configs             one per cluster type: expected agents (shape assumed)
    static API documents - NOT written by default; switch one on with "collections": {"<kind>": "<collection name>"}
    (collection then holds only this org's document(s); anything else needs --adopt)
        static_config, static_gpu_efficiency_trend (1/7/15/30 d), static_nodeagent_health, static_neocloud_overview,
        static_resources, static_rate_card, static_workload_health

Safety: nothing is written unless the renter engine verification AND the document checks pass; protected hosts
(MOCK_ORG_PROTECTED_HOSTS, default "prod,production") are refused; every document that is replaced or deleted is backed up
to ~/.mock-org-backups/<db>/<time>/; a manifest in <db>._mock_org_tool remembers what this tool wrote, so a re-run removes
only its own stale documents.
"""

import argparse
import base64
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from pathlib import Path
from urllib.parse import quote_plus

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import renter_spec as rs  # noqa: E402  (stdlib only; loads engine/ lazily)

OUT_ROOT = HERE / "orgs" / "renter"
MANIFEST_COLL = rs.MARKER_COLLECTION                     # "_mock_org_tool"
INPUT_KEYS = set(rs.RENTER_KEYS) | {"mongo", "collections", "_comment"}

DEFAULT_COLLECTIONS = OrderedDict([
    ("entities", "caladrius_neo_cloud_entities"),
    ("gpu_metrics", "caladrius_neo_cloud_gpu_metrics"),
    ("workload_health", "caladrius_neo_cloud_workload_health_snapshots"),   # one per workload (the static_workload_health
                                                                              # summary document is the kind further down)
    ("rate_cards", "caladrius_neo_cloud_rate_cards"),
    ("resources", "self_managed_resources"),
    ("project_configs", "project_configs"),
    # Settings > Integrations, shaped exactly as wrench-service reads them (neo_cloud_install_command_service.go,
    # neo_cloud_install_cache.go, neo_cloud_exporter_health_service.go). delegate_install_templates and
    # gpu_exporter_agent_health_configs live in caladrius_master (shared by all orgs) and are NOT written.
    ("monitored_entities", "caladrius_monitored_entities"),      # one per cluster: attributes.id = resourceId, isSelected
    ("delegate_configs", "delegate_configs"),                    # one per cluster: resourceConfig, agenticLastSyncAt,
                                                                 # nodeAgentConfigs (CONNECTED / healthy if < 5 min old)
    # static API documents. TEMPORARILY ON: the backend reads these first (Integrations, workload health, GPU fleet,
    # efficiency trend, thresholds) - same set as the Renter-Demo org. Turn one off with "collections": {"<kind>": null}
    # in the input once the backend reads the original collections above.
    ("static_resources", "static_resources"),
    ("static_nodeagent_health", "static_nodeagent_health"),
    ("static_config", "static_config"),
    ("static_gpu_efficiency_trend", "static_gpu_efficiency_trend"),
    ("static_neocloud_overview", "static_neocloud_overview"),
    ("static_rate_card", "static_rate_card"),
    ("static_workload_health", "static_workload_health"),
])
STATIC_FILES = OrderedDict([("static_config", "config.json"), ("static_nodeagent_health", "nodeagent-health.json"),
                            ("static_neocloud_overview", "overview.json"), ("static_resources", "resources.json"),
                            ("static_rate_card", "rate-card.json"), ("static_workload_health", "workload-health.json")])
TREND_DAYS = (1, 7, 15, 30)
REPLACE_KINDS = set(STATIC_FILES) | {"static_gpu_efficiency_trend"}   # the UI reads the single document of these

# entity "type" values (as wrench-service reads them)
T_NODE, T_GPU, T_FS = "GPU_INSTANCE", "GPU_DEVICE", "FILESYSTEM"   # cluster / workload types: see CLUSTER_TYPE, w["kind"]
OWNER_TOOL = "GRAFANA"
ZERO_PROJECT = "000000000000000000000000"
HISTORY_DAYS = 30


class RenterError(Exception):
    pass


# ---------------------------------------------------------------- small helpers
def hid(*parts, n=24):
    """Deterministic hex id: same org + entity -> same id, so a re-run replaces instead of duplicating."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:n]


def nl(v):
    return {"$numberLong": str(int(v))}                  # Extended JSON -> Int64


def oid(h):
    return {"$oid": h}                                   # Extended JSON -> ObjectId


def raw_id(d):
    return d["_id"]["$oid"] if isinstance(d["_id"], dict) else d["_id"]


def id_key(d):
    return ("o:" if isinstance(d["_id"], dict) else "s:") + raw_id(d)


def rel(kind, target, ttype):
    return {"type": kind, "targetCanonical": target, "targetId": target, "targetType": ttype}


def drop(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def chunks(xs, n=1000):
    for k in range(0, len(xs), n):
        yield xs[k:k + n]


# ---------------------------------------------------------------- input
def load_input(path):
    try:
        req = rs.load_json(path)
    except (OSError, ValueError) as e:
        raise RenterError(f"cannot read {path}: {e}")
    if not isinstance(req, dict):
        raise RenterError(f"{path}: the input must be a JSON object")
    if str(req.get("orgType", "RENTER")).upper() != "RENTER":
        raise RenterError("this tool builds RENTER orgs only (orgType must be RENTER or absent)")
    extra = sorted(set(req) - INPUT_KEYS)
    if extra:
        raise RenterError(f"unknown key(s) {extra}; allowed: {sorted(INPUT_KEYS - {'_comment'})}")
    org = {k: v for k, v in (req.get("org") or {}).items()           # None, "" and "<placeholder>" = not given
           if v not in (None, "") and not (isinstance(v, str) and "<" in v)}
    if not (org.get("orgName") or org.get("name")):
        raise RenterError("org.orgName is required")
    mongo = req.get("mongo") or {}
    rs.check_keys(mongo, {"uri", "database"}, "mongo")
    colls = OrderedDict(DEFAULT_COLLECTIONS)
    over = req.get("collections") or {}
    rs.check_keys(over, set(DEFAULT_COLLECTIONS), "collections")
    colls.update(over)
    names = {}
    for kind, name in colls.items():
        if name is None:
            continue
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,119}", name) or name == MANIFEST_COLL:
            raise RenterError(f"collections.{kind}: invalid collection name {name!r}")
        names.setdefault(name, []).append(kind)
    for name, kinds in names.items():
        if len(kinds) > 1 and set(kinds) & REPLACE_KINDS:
            raise RenterError(f"collection {name!r} is used by {kinds}; a static collection cannot be shared")
    # engine request: only the renter engine keys (no uri / database ends up in request.json on disk)
    engine_req = {k: v for k, v in req.items() if k in rs.RENTER_KEYS and k not in ("target", "orgType")}
    engine_req["orgType"] = "RENTER"
    engine_req["org"] = org
    return engine_req, mongo, colls, req.get("target") or {}


# ---------------------------------------------------------------- documents
def entity_factory(org_id, snap_s, ents):
    tool_key = hid(org_id, "tool", OWNER_TOOL, n=64)
    tool_oid = hid(org_id, "tool-id", OWNER_TOOL)
    first_ms, snap_ms = (snap_s - HISTORY_DAYS * 86400) * 1000, snap_s * 1000
    tool = OWNER_TOOL.lower()

    def entity(etype, name, cluster, attrs, rels, subcat=None, category="INFRA"):
        keys = ([f"{tool}:{etype.lower()}:{cluster}:{name}"] if cluster else []) + [f"{tool}:{etype.lower()}:{name}"]
        ents.append({"_id": hid(org_id, "entity", etype, name), "attributes": attrs, "canonicalName": name, "category": category,
                     "cluster": cluster, "createdAt": nl(first_ms), "displayName": name, "firstSeen": nl(first_ms),
                     "instance": "", "isDeleted": False, "lastSeen": nl(snap_ms), "matchingKeys": keys, "orgId": org_id,
                     "ownerTool": OWNER_TOOL, "ownerToolKey": tool_key, "projectId": ZERO_PROJECT, "relationships": rels,
                     "sources": [{"tool": OWNER_TOOL, "toolId": oid(tool_oid), "toolKey": tool_key, "entityId": name}],
                     "subcategory": subcat or etype, "tags": {}, "type": etype, "updatedAt": nl(snap_ms)})
    return entity


def filesystem_attrs(seed, name, node, cap, used, fs_type, mount):
    r = rs.rng_for(seed, "filesystem", name)
    return {"node": node, "probe_success": 1, "used_gib": used, "util_pct": round(used / cap * 100.0, 1),
            "write_latency_s": round(r.uniform(0.002, 0.005), 6), "free_gib": round(cap - used, 1), "fs_type": fs_type,
            "fsync_latency_s": round(r.uniform(0.001, 0.002), 6), "read_latency_s": round(r.uniform(0.001, 0.002), 6),
            "capacity_gib": cap, "mount_path": mount}


def health_snapshot(org_id, workload_id, name, kind, infra, namespace, container, cluster, cats, overall, computed):
    """caladrius_neo_cloud_workload_health_snapshots row as WorkloadHealthSnapshot decodes it: ObjectId _id, workload_id =
    the workload entity _id (the join key), infra_source K8s / SLURM / Standalone, computed_at a date."""
    iso = re.sub(r"(\.\d{3})\d*Z$", r"\1Z", computed)              # nanoseconds -> milliseconds for $date
    return {"_id": oid(hid(org_id, "workload-health", name)), "orgId": org_id, "workload_id": workload_id,
            "workload_name": name, "workload_type": kind, "infra_source": infra, "namespace": namespace,
            "container_name": container, "cluster": cluster, "overall_health": overall, "data_error": "",
            "categories": [{"category_id": c["categoryId"], "category_name": c["categoryName"], "health": c["health"],
                            "weighted_score": c["weightedScore"], "effective_weight": c["effectiveWeight"],
                            "critical_override": c["criticalOverride"],
                            "metrics": [{"metric_id": m["metricId"], "metric_name": m["metricName"],
                                         "current_value": m["currentValue"], "unit": m["unit"], "health": m["health"].lower(),
                                         "is_critical": m["isCritical"], "weight": m["weight"], "score": m["score"]}
                                        for m in c["metrics"]]} for c in cats],
            "computed_at": {"$date": iso}}


# GPU-metric history: the overview efficiency widget averages each GPU over the configured window (default 30 days) and the
# trend endpoint buckets the same documents per UTC day. History stays inside the GPU's band, so the band split equals
# the engine's (an average of values inside an interval stays inside it); the newest document is the current reading.
METRIC_HISTORY_DAYS = 30
BAND_RANGE = {"idle": (0.0, 0.0), "under": (1.0, 19.5), "normal": (21.5, 79.5), "over": (81.5, 99.5)}


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def gpu_metrics_docs(org_id, g, band, snap_s, seed, ranges, overridden=()):
    """Current reading + one per past day. Past days stay in the GPU's band and inside the cluster's idle / busy
    temperature, power and memory ranges; temperature and power follow utilisation the way the engine models them."""
    def doc(day, util, mem, temp, power):
        return {"_id": oid(hid(org_id, "gpu-metrics", g["name"], *([day] if day else []))), "orgId": org_id,
                "entityType": T_GPU, "entityName": g["name"], "cluster": g["cluster"],
                "metrics": [{"dataType": "utilization_pct", "data": util}, {"dataType": "memory_used_mib", "data": mem},
                            {"dataType": "memory_total_mib", "data": g["memoryTotalMib"]},
                            {"dataType": "temperature_c", "data": temp}, {"dataType": "power_draw_w", "data": power}],
                "createdAt": nl((snap_s - day * 86400) * 1000)}
    util, mem, temp, power = g["utilizationPct"], g["memoryUsedMib"], g["temperatureC"], g["powerDrawW"]
    out = [doc(0, util, round(mem), temp, power)]
    lo, hi = BAND_RANGE[band]
    state = "idle" if band == "idle" else "busy"
    t_r, p_r, m_r = (ranges[m][state] for m in ("temperatureC", "powerW", "memoryUsedPct"))
    total = g["memoryTotalMib"]
    m_lo, m_hi = int(-(-m_r[0] * total // 100)), int(m_r[1] * total // 100)      # whole MiB inside the % range
    r = rs.rng_for(seed, "metric-history", g["name"])
    for day in range(1, METRIC_HISTORY_DAYS):
        u = 0.0 if band == "idle" else round(clip(util + r.gauss(0.0, (hi - lo) * 0.12), lo, hi), 1)
        jitter = r.uniform(0.95, 1.05)
        t = temp if "temperatureC" in overridden else round(clip(temp + (u - util) * (t_r[1] - t_r[0]) / 100, *t_r), 1)
        p = power if "powerDrawW" in overridden else round(clip(power + (u - util) * (p_r[1] - p_r[0]) / 100, *p_r), 1)
        m = round(mem) if "memoryUsedMib" in overridden else int(clip(round(mem * jitter), m_lo, m_hi))
        out.append(doc(day, u, m, t, p))
    return out


# entity model read by wrench-service neo_cloud_overview_service.go / neo_cloud_workload_health_service.go
CLUSTER_TYPE = {"slurm": "SLURMCLUSTER", "k8s": "KUBERNETESCLUSTER"}     # GPU_CLUSTER is never queried
DEVICE_PARENT = {"slurm": "HOSTED_ON", "k8s": "BELONGS_TO"}               # GPU_DEVICE -> its GPU_INSTANCE
SNAPSHOT_INFRA = {"slurm": "SLURM", "k8s": "K8s"}
T_POD, T_JOB = "POD", "SLURM_JOB"


def renter_entities(org_id, snap_s, seed, inv, ov, wh, out, overridden=None):
    """caladrius_neo_cloud_entities / _gpu_metrics / _workload_health_snapshots. Returns the expected count per type."""
    overridden = overridden or {}
    ranges_of = {c["name"]: c["metricRanges"] for c in inv["clusters"]}
    gpu_of = {c["name"]: c["gpu"] for c in inv["clusters"]}
    entity = entity_factory(org_id, snap_s, out["entities"])
    devs = {g["name"]: g for g in ov["gpuDevices"]}
    ov_clusters = {c["name"]: c for c in ov["clusters"]}
    wh_items = {w["workloadName"]: w for blk in ("training", "inference") for w in wh[blk]["workloads"]}
    inv_gpus = {g["name"]: g for g in inv["gpus"]}
    kind_of = {c["name"]: c["kind"] for c in inv["clusters"]}
    nodes_of = {}
    for n in inv["nodes"]:
        nodes_of.setdefault(n["cluster"], []).append(n["name"])
    in_cluster = lambda cl: rel("IN_CLUSTER", cl, CLUSTER_TYPE[kind_of[cl]])         # noqa: E731
    first_ms = (snap_s - HISTORY_DAYS * 86400) * 1000

    for c in inv["clusters"]:
        name, ctype = c["name"], CLUSTER_TYPE[c["kind"]]
        model = devs[next(g for g in inv["gpus"] if g["cluster"] == name)["name"]]["model"]
        entity(ctype, name, name, {
            "name": name, "alias": c["alias"], "kind": c["kind"], "infra": c["infra"], "status": "healthy",
            "nodeCount": c["nodeCount"], "gpuCount": c["gpuCount"], "workloadCount": len(c["workloads"]),
            "avgUtilizationPct": ov_clusters.get(name, {}).get("avgUtilizationPct"), "gpuModel": model,
            "gpuType": gpu_of[name]["type"], "gpusPerNode": gpu_of[name]["gpusPerNode"]}, [])
    for n in inv["nodes"]:
        gs = [devs[g] for g in n["gpus"]]
        entity(T_NODE, n["name"], n["cluster"], {
            "node": n["name"], "rack": n["rack"], "index": n["index"], "infra": gs[0]["infra"], "status": "healthy",
            "model": gs[0]["model"], "gpu_count": len(gs), "gpuCount": len(gs), "gpuModel": gs[0]["model"],
            "avgUtilizationPct": round(mean(g["utilizationPct"] for g in gs), 1),
            "avgTemperatureC": round(mean(g["temperatureC"] for g in gs), 1),
            "totalMemoryUsedMib": round(sum(g["memoryUsedMib"] for g in gs), 1), "totalMemoryMib": sum(g["memoryTotalMib"] for g in gs),
            "totalPowerW": round(sum(g["powerDrawW"] for g in gs), 1), "storage": n["storage"]},
            [in_cluster(n["cluster"])])
        for g in gs:
            ig = inv_gpus[g["name"]]
            entity(T_GPU, g["name"], g["cluster"], {
                **drop(g, "name", "parent", "cluster", "gpuIndex"), "gpu_index": g["gpuIndex"], "index": g["gpuIndex"],
                "model": g["model"], "memory_total_mib": g["memoryTotalMib"], "node": n["name"], "status": "healthy",
                "band": ig["band"], "workload": ig["workload"]},
                [in_cluster(g["cluster"]), rel(DEVICE_PARENT[kind_of[g["cluster"]]], n["name"], T_NODE)])
            out["gpu_metrics"].extend(gpu_metrics_docs(org_id, g, ig["band"], snap_s, seed, ranges_of[g["cluster"]],
                                                       overridden.get(g["name"], ())))
    n_pods = n_jobs = 0
    for w in inv["workloads"]:
        h, cl = wh_items[w["name"]], w["cluster"]
        kind = kind_of[cl]
        by_node = OrderedDict()
        for gid in w["gpus"]:
            by_node.setdefault(inv_gpus[gid]["node"], []).append(gid)
        attrs = {"workloadName": w["name"], "workloadType": w["kind"], "health": h["overallHealth"],
                 "namespace": h["namespace"], "containerName": h["containerName"], "infraSource": SNAPSHOT_INFRA[kind],
                 "gpuCount": w["gpuCount"], "nodeCount": len(by_node), "nodes": list(by_node), "gpus": w["gpus"],
                 "start_time": first_ms}
        if kind == "k8s":                    # WORKLOAD -> POD (one per node) -> GPU_DEVICE
            pods = [f"{w['name']}-pod-{i:02d}" for i in range(len(by_node))]
            for pod, (node, gids) in zip(pods, by_node.items()):
                entity(T_POD, pod, cl, {"namespace": h["namespace"], "node": node, "workload": w["name"]},
                       [in_cluster(cl), rel("RUNS_ON", node, T_NODE)] + [rel("USES_GPU", gid, T_GPU) for gid in gids])
            n_pods += len(pods)
            links = [rel("HAS_POD", pod, T_POD) for pod in pods]
        else:                                # WORKLOAD -> SLURMCLUSTER + SLURM_JOB -> GPU_DEVICE
            job = f"{w['name']}-job"
            entity(T_JOB, job, cl, {"workload": w["name"], "nodes": list(by_node)},
                   [in_cluster(cl)] + [rel("RUNS_ON", nd, T_NODE) for nd in by_node] +
                   [rel("USES_GPU", gid, T_GPU) for gid in w["gpus"]])
            n_jobs += 1
            links = [rel("HAS_JOB", job, T_JOB)]
        entity(w["kind"], w["name"], cl, attrs, [in_cluster(cl)] + links, subcat=w["kind"], category="WORKLOAD")
        out["workload_health"].append(health_snapshot(
            org_id, hid(org_id, "entity", w["kind"], w["name"]), w["name"], w["kind"], SNAPSHOT_INFRA[kind],
            h["namespace"], h["containerName"], cl, h["categories"], h["overallHealth"], h["computedAt"]))
    for f in inv["storage"]:
        on = [f["node"]] if f["node"] else nodes_of[f["cluster"]]
        entity(T_FS, f["name"], f["cluster"], {**filesystem_attrs(seed, f["name"], on[0], f["capacityGib"], f["usedGib"], f["fsType"],
                                                                  "/mnt/" + f["name"]), "kind": f["kind"]},
               [rel("MOUNTED_ON", nid, T_NODE) for nid in on])
    kinds = [c["kind"] for c in inv["clusters"]]
    wl = Counter(w["kind"] for w in inv["workloads"])
    t = inv["totals"]
    return {"SLURMCLUSTER": kinds.count("slurm"), "KUBERNETESCLUSTER": kinds.count("k8s"), T_NODE: t["nodes"],
            T_GPU: t["gpus"], "TRAINING_JOB": wl.get("TRAINING_JOB", 0), "INFERENCE_SERVICE": wl.get("INFERENCE_SERVICE", 0),
            T_FS: t["storage"], T_POD: n_pods, T_JOB: n_jobs}


def renter_docs(work, org, snap_s, seed):
    """Every document kind for one renter org, from the engine output in <work>/generated (already verified)."""
    gen = work / "generated"
    L = lambda n: rs.load_json(gen / n)                                          # noqa: E731
    inv, ov, wh = L("inventory.json"), L("overview.json")["data"], L("workload-health.json")["data"]
    org_id = org["orgId"]
    out = OrderedDict((k, []) for k in DEFAULT_COLLECTIONS)
    want_types = renter_entities(org_id, snap_s, seed, inv, ov, wh, out, rs.load_json(work / "expect.json")["overridden"])
    # resources + project configs keep the engine ids: static_resources, self_managed_resources.projectId and
    # project_configs._id must point at each other for the UI.
    out["resources"] = L("resources-mongo.json")
    out["project_configs"] = L("project_configs.json")
    for d in L("rate-card.json")["data"]:
        out["rate_cards"].append({"_id": oid(hid(org_id, "rate-card", d["clusterName"])), **d, "updatedAt": nl(d["updatedAt"])})
    for kind, fname in STATIC_FILES.items():
        out[kind] = [{"_id": oid(hid(org_id, kind)), "data": L(fname)["data"]}]
    for days in TREND_DAYS:
        data = L(f"gpu-efficiency-trend-{days}d.json")["data"]
        out["static_gpu_efficiency_trend"].append({"_id": oid(hid(org_id, "static_gpu_efficiency_trend", days)),
                                                   "days": data["durationDays"], "data": data})

    delegate_docs(org, L("resources.json")["data"], L("nodeagent-health.json")["data"], out, snap_s)
    check_metric_ranges(out["gpu_metrics"], inv, rs.load_json(work / "expect.json")["overridden"])

    t = inv["totals"]
    want_docs = {"gpu_metrics": t["gpus"] * METRIC_HISTORY_DAYS, "resources": t["clusters"], "rate_cards": t["clusters"],
                 "workload_health": t["workloads"], "project_configs": t["clusters"],
                 "static_gpu_efficiency_trend": len(TREND_DAYS), **{k: 1 for k in STATIC_FILES},
                 "monitored_entities": t["clusters"], "delegate_configs": t["clusters"]}
    return out, want_types, want_docs


# ---------------------------------------------------------------- delegate / agent collections (Settings > Integrations)
# Shapes follow wrench-service + caladrius-commons-libs (DelegateConfig, CaladriusMonitoredEntity) and the documents
# AddSelfManagedResources creates for a real resource:
#   GET /self-managed/resources   delegate_configs{orgId, isDeleted:false, resourceConfig.id = resourceId}
#                                 + caladrius_monitored_entities{isDeleted:false, isSelected:true, attributes.id = resourceId}
#                                 + caladrius_master.delegate_install_templates{resourceType}
#                                 agentStatus = CONNECTED if now - agenticLastSyncAt (seconds) <= 300
#   GET /exporter-agent-health    delegate_configs.agenticLastSyncAt (s) + nodeAgentConfigs.<agent>.lastSyncAt (ms) <= 5 min
#                                 for every agent in caladrius_master.gpu_exporter_agent_health_configs{resourceType}
DIRECT_REFERENCE = "Direct Connect"                 # constants.CALADRIUS_DIRECT_REFERENCE_NAME
TYPE_SUBCATEGORY = {"standalone": "GPU_INSTANCE", "kubernetes_cluster": "KUBERNETES_CLUSTER", "slurm_cluster": "SLURM_CLUSTER"}
# node agents stamped in nodeAgentConfigs; the upload adds every agent the master config names for the resource type
DEFAULT_NODE_AGENTS = ["topology_exporter", "node_exporter", "filesystem_exporter", "rdma_exporter", "dcgm_exporter",
                       "log_exporter"]


def delegate_docs(org, res, na, out, snap_s):
    """delegate_configs + caladrius_monitored_entities, one each per cluster, from the engine's resources /
    nodeagent-health documents (same delegate keys, resource ids and sync times as static_resources)."""
    org_id = org["orgId"]
    smr = {d["name"]: d for d in out["resources"]}
    health = {a["resourceConfigName"]: a for a in na}
    for r in res:
        name, rtype = r["name"], r["type"]["value"]
        h = health[name]
        if h["delegateKey"] != r["delegateKey"]:
            raise RenterError(f"{name}: delegate key differs between resources and agent health in the engine output")
        tool = r["provider"]["value"].upper()                   # ensureProviderToolConfig: strings.ToUpper(provider)
        sub = TYPE_SUBCATEGORY.get(rtype, rtype.upper())
        created = int(smr[name]["createdAt"]["$numberLong"])
        sync_s = int(r["agenticLastSyncAt"])
        sync_ms = max(int(a["lastSyncAt"]) for a in h["agents"])
        stamp = {"createdAt": nl(created), "updatedAt": nl(snap_s), "createdBy": org["createdBy"], "updatedBy": org["updatedBy"]}
        out["delegate_configs"].append({
            "_id": oid(hid(org_id, "delegate-config", name)), "orgId": org_id, "orgName": org["orgName"],
            "orgSlug": org["orgSlug"], "referenceName": f"{name}-agent", "isHosted": True, "delegateKey": r["delegateKey"],
            "isDeleted": False, "isDraft": False, "isEnabled": True, "isUpdateAvailable": False,
            "installationExpiryTime": nl(0), "lifecycleExpiryTime": nl(0), "version": "", "ConfigVersion": "",
            "lastSyncAt": nl(sync_s), "agenticLastSyncAt": nl(sync_s), "tool": tool,
            "resourceConfig": {"id": r["resourceId"], "name": name, "region": r["region"], "type": rtype},
            "nodeAgentConfigs": {a: {"lastSyncAt": nl(sync_ms), "enabled": True} for a in DEFAULT_NODE_AGENTS}, **stamp})
        out["monitored_entities"].append({
            "_id": oid(hid(org_id, "monitored", name)), "orgId": org_id, "orgName": org["orgName"], "tool": tool,
            "toolKey": r["toolKey"], "toolConfigId": oid(hid(org_id, "tool-config", tool)),
            "projectId": oid(r["projectId"]), "projectName": r["projectName"],
            "dataSourceReferenceName": f"{DIRECT_REFERENCE} - {r['provider']['label']}", "dataSourceTypes": [],
            "matchingKey": f"{tool}:{rtype}:{r['resourceId']}".lower(), "canonicalName": name, "category": "INFRA",
            "subcategory": sub, "type": sub, "namespace": "", "hierarchyLevel": 0, "hierarchyPath": [],
            "attributes": {"details": "", "id": r["resourceId"], "region": r["region"]},
            "isSelected": True, "selectionStatus": "LINKED", "isDeleted": False, **stamp})


def check_metric_ranges(docs, inv, overridden):
    """Every GPU-metric document (today and the 29 days before) inside its cluster's idle / busy ranges."""
    ranges = {c["name"]: c["metricRanges"] for c in inv["clusters"]}
    field = {"temperature_c": ("temperatureC", "temperatureC"), "power_draw_w": ("powerW", "powerDrawW")}
    errs = []
    for d in docs:
        m = {x["dataType"]: x["data"] for x in d["metrics"]}
        state = "idle" if m["utilization_pct"] == 0 else "busy"
        rg, ovr = ranges[d["cluster"]], overridden.get(d["entityName"], ())
        for dt_, (key, src) in field.items():
            lo, hi = rg[key][state]
            if src not in ovr and not lo - 0.1 <= m[dt_] <= hi + 0.1:
                errs.append(f"{d['entityName']}: {dt_} {m[dt_]} outside {state} range [{lo}, {hi}]")
        pct = m["memory_used_mib"] / m["memory_total_mib"] * 100
        lo, hi = rg["memoryUsedPct"][state]
        if "memoryUsedMib" not in ovr and not lo - 0.1 <= pct <= hi + 0.1:
            errs.append(f"{d['entityName']}: memory {pct:.1f}% outside {state} range [{lo}, {hi}]")
    if errs:
        raise RenterError(f"GPU metric documents outside the requested ranges ({len(errs)}): " + "; ".join(errs[:8]))


def check_docs(kinds, want_types, want_docs, org_id):
    """Counts equal the generated fleet, ids are unique, relationships resolve, cross-collection ids agree, one orgId."""
    errs = []
    ents = kinds["entities"]
    got = Counter(e["type"] for e in ents)
    for t, n in want_types.items():
        if got.get(t, 0) != n:
            errs.append(f"{t}: {got.get(t, 0)} entities, expected {n}")
    for k, n in want_docs.items():
        if len(kinds[k]) != n:
            errs.append(f"{k}: {len(kinds[k])} documents, expected {n}")
    for k, docs in kinds.items():
        if len({id_key(d) for d in docs}) != len(docs):
            errs.append(f"{k}: duplicate _id")
        for d in docs:
            if "orgId" in d and d["orgId"] != org_id:
                errs.append(f"{k}: document {raw_id(d)} has orgId {d['orgId']}, expected {org_id}")
                break
    types = {}
    for e in ents:
        if e["canonicalName"] in types:
            errs.append(f"canonicalName {e['canonicalName']} used twice")
        types[e["canonicalName"]] = e["type"]
    for e in ents:
        for r in e["relationships"]:
            if types.get(r["targetId"]) != r["targetType"]:
                errs.append(f"{e['canonicalName']}: {r['type']} -> {r['targetId']} ({r['targetType']}) does not exist")
    projects = {raw_id(p): p["projectName"] for p in kinds["project_configs"]}
    for r in kinds["resources"]:
        pid = r["projectId"]["$oid"] if isinstance(r["projectId"], dict) else r["projectId"]
        if projects.get(pid) != r["name"]:
            errs.append(f"self_managed_resources {r['name']}: projectId {pid} is not the project of that cluster")
    static_names = {r["name"] for r in kinds["static_resources"][0]["data"]} if kinds["static_resources"] else set()
    if static_names != {r["name"] for r in kinds["resources"]}:
        errs.append(f"static_resources clusters {sorted(static_names)} != self_managed_resources {sorted(r['name'] for r in kinds['resources'])}")
    # Settings > Integrations - the joins wrench-service makes (see delegate_docs): per self_managed_resources document
    # one delegate_configs with resourceConfig.id = resourceId and one caladrius_monitored_entities with
    # attributes.id = resourceId (isSelected, not deleted); same orgId, type, region, projectId; sync times set
    smr = {r["resourceId"]: r for r in kinds["resources"]}
    dc = {d["resourceConfig"]["id"]: d for d in kinds["delegate_configs"]}
    me = {m["attributes"]["id"]: m for m in kinds["monitored_entities"]}
    for rid, r in smr.items():
        d, m, name = dc.get(rid), me.get(rid), r["name"]
        if not d or not m:
            errs.append(f"cluster {name}: no delegate_configs (resourceConfig.id) or caladrius_monitored_entities "
                        f"(attributes.id) for resourceId {rid[:12]} - the UI shows NOT_INSTALLED")
            continue
        rc = d["resourceConfig"]
        if (rc["name"], rc["type"], rc["region"]) != (r["name"], r["type"], r["region"]):
            errs.append(f"delegate_configs {name}: resourceConfig name/type/region differ from self_managed_resources")
        if d["isDeleted"] or not d["isEnabled"] or d["orgId"] != r["orgId"] or not d["delegateKey"]:
            errs.append(f"delegate_configs {name}: deleted, disabled, wrong orgId or no delegateKey")
        if int(d["agenticLastSyncAt"]["$numberLong"]) <= 0 or not d["nodeAgentConfigs"]:
            errs.append(f"delegate_configs {name}: agenticLastSyncAt / nodeAgentConfigs not set - agent never seen")
        if m["isDeleted"] or not m["isSelected"] or m["projectId"]["$oid"] != r["projectId"]["$oid"]:
            errs.append(f"caladrius_monitored_entities {name}: deleted, not selected or projectId differs")
    for extra in (set(dc) | set(me)) - set(smr):
        errs.append(f"delegate / monitored document for unknown resourceId {extra[:12]}")
    if errs:
        raise RenterError(f"document check failed ({len(errs)}): " + "; ".join(errs[:10]))


def to_collections(kinds, names):
    colls, modes = OrderedDict(), {}
    for kind, coll in names.items():
        if coll:
            colls.setdefault(coll, []).extend(kinds[kind])
            modes[coll] = "replace" if kind in REPLACE_KINDS else "upsert"
    return colls, modes


def requirement_check(work):
    """Print what was built per cluster next to what was asked (ranges) and re-check it against the inventory."""
    inv = rs.load_json(work / "generated" / "inventory.json")
    expect = rs.load_json(work / "expect.json")
    built = {c["name"]: c for c in inv["clusters"]}
    kinds = Counter((w["cluster"], w["kind"]) for w in inv["workloads"])
    fmt = lambda r: "" if r is None else f"{r[0]}" if r[0] == r[1] else f"{r[0]}-{r[1]}"       # noqa: E731
    print("\n  requirement check (built / asked)")
    print(f"    {'CLUSTER':14} {'KIND':6} {'GPU':7} {'NODES':>10} {'GPUS':>5} {'TRAINING':>10} {'INFERENCE':>10}  TEMP C / POWER W (busy)")
    errs = []
    for e in expect["clusters"]:
        b, wa = built[e["name"]], e["workloadsAsked"] or {}
        t, i = kinds[(e["name"], "TRAINING_JOB")], kinds[(e["name"], "INFERENCE_SERVICE")]
        rg = e["ranges"]
        print(f"    {e['name']:14} {e['kind']:6} {e['gpu']['type']:7} {b['nodeCount']:>3} / {fmt(e['nodesAsked']):<5} "
              f"{b['gpuCount']:>5} {t:>3} / {fmt(wa.get('TRAINING_JOB')):<4} {i:>3} / {fmt(wa.get('INFERENCE_SERVICE')):<4}  "
              f"{rg['temperatureC']['busy'][0]:g}-{rg['temperatureC']['busy'][1]:g} / {rg['powerW']['busy'][0]:g}-{rg['powerW']['busy'][1]:g}")
        if b["nodeCount"] != e["nodes"] or b["gpuCount"] != e["nodes"] * e["gpu"]["gpusPerNode"]:
            errs.append(f"cluster {e['name']}: built {b['nodeCount']} nodes / {b['gpuCount']} GPUs")
        if (t, i) != (e["workloads"]["TRAINING_JOB"], e["workloads"]["INFERENCE_SERVICE"]):
            errs.append(f"cluster {e['name']}: built {t} training / {i} inference")
    if errs:
        raise RenterError("generated org does not match the input: " + "; ".join(errs))
    print("    every cluster matches its request; metric values are inside their ranges")


# ---------------------------------------------------------------- upload
def key_of(raw):
    from bson import ObjectId
    return ("o:" + str(raw)) if isinstance(raw, ObjectId) else ("s:" + str(raw))


def from_key(k):
    from bson import ObjectId
    return ObjectId(k[2:]) if k.startswith("o:") else k[2:]


def add_node_agents(docs, names, rtype=None):
    """Stamp agent names into nodeAgentConfigs (same time as the existing agents) for all delegates, or one type."""
    for d in docs:
        if rtype is None or d["resourceConfig"]["type"] == rtype:
            ms = next(iter(d["nodeAgentConfigs"].values()))["lastSyncAt"]
            for n in names:
                d["nodeAgentConfigs"].setdefault(n, {"lastSyncAt": ms, "enabled": True})


def master_agent_check(client, master_db, colls, delegate_coll, manual=()):
    """Read-only look at caladrius_master: add every agent gpu_exporter_agent_health_configs requires for a resource type
    to nodeAgentConfigs (else Node Agents read 'never_seen' / not installed), and warn when delegate_install_templates
    has no template for a type (the backend then skips the resource -> NOT_INSTALLED). `manual` (--node-agents) is
    added to every delegate as well."""
    docs = colls.get(delegate_coll) or []
    if not docs:
        return
    if manual:
        add_node_agents(docs, manual)
        print(f"  node agents from --node-agents: {list(manual)}")
    mdb = client[master_db]
    for rtype in sorted({d["resourceConfig"]["type"] for d in docs}):
        try:
            cfg = mdb["gpu_exporter_agent_health_configs"].find_one({"resourceType": rtype, "isActive": True})
            tpl = mdb["delegate_install_templates"].find_one({"resourceType": rtype}, {"_id": 1})
        except Exception as e:                                                  # noqa: BLE001 - no master access is not fatal
            print(f"\n  WARNING: cannot read {master_db}.gpu_exporter_agent_health_configs ({type(e).__name__}: "
                  f"{str(e).split(', full error')[0][:120]}).\n  The backend requires every agent listed there for "
                  f"{rtype}; any name missing from nodeAgentConfigs shows Node Agents 'not installed' and the cluster "
                  f"'not connected'.\n  Fix: pass the names with --node-agents (ask for the mandatoryAgents + "
                  f"optionalAgents of {rtype}), or use a login that can read {master_db}.\n")
            continue
        if not tpl:
            print(f"  WARNING: {master_db}.delegate_install_templates has no template for {rtype}; the backend skips "
                  f"these resources (NOT_INSTALLED) until one exists")
        if not cfg:
            print(f"  WARNING: {master_db}.gpu_exporter_agent_health_configs has no active (isActive: true) config for "
                  f"{rtype}; the backend skips these delegates in agent health")
            continue
        names = list(cfg.get("mandatoryAgents") or []) + list(cfg.get("optionalAgents") or [])
        add_node_agents(docs, names, rtype)
        print(f"  {master_db}: {rtype} needs agents {names} - all stamped" + ("" if tpl else " (no install template)"))


def stamp_agents_now(colls, delegate_coll, now_s):
    """Delegate / node-agent sync times = the moment of writing: the backend shows CONNECTED / healthy for 5 minutes."""
    for d in colls.get(delegate_coll) or []:
        d["agenticLastSyncAt"] = d["lastSyncAt"] = nl(now_s)
        for a in d["nodeAgentConfigs"].values():
            a["lastSyncAt"] = nl(now_s * 1000)


def heartbeat(db, delegate_coll, ids, every_s):
    """Keep this org's delegates fresh (CONNECTED / healthy) until Ctrl+C, like a real agent reporting in."""
    from bson import ObjectId
    oids = [ObjectId(i) for i in ids]
    print(f"\n  heartbeat: refreshing {len(oids)} delegate(s) every {every_s} s - press Ctrl+C to stop "
          f"(the UI shows DISCONNECTED about 5 minutes after it stops)")
    try:
        while True:
            now = int(time.time())
            for d in db[delegate_coll].find({"_id": {"$in": oids}}, {"nodeAgentConfigs": 1}):
                upd = {"agenticLastSyncAt": now, "lastSyncAt": now, "updatedAt": now}
                upd.update({f"nodeAgentConfigs.{a}.lastSyncAt": now * 1000 for a in (d.get("nodeAgentConfigs") or {})})
                db[delegate_coll].update_one({"_id": d["_id"]}, {"$set": upd})
            print(f"  heartbeat {time.strftime('%H:%M:%S')}  {len(oids)} delegate(s) stamped")
            time.sleep(every_s)
    except KeyboardInterrupt:
        print("  heartbeat stopped")


def upload(colls, modes, uri, db_name, org, meta, backup_dir, adopt, yes, master_db="caladrius_master",
           delegate_coll=None, beat=0, ahead_s=0, node_agents=()):
    from bson import json_util
    from pymongo import ReplaceOne

    host = rs.host_of(uri).lower()
    protected = [p.strip().lower() for p in os.environ.get("MOCK_ORG_PROTECTED_HOSTS", "prod,production").split(",") if p.strip()]
    if any(p in host for p in protected):
        raise RenterError(f"host {host} matches a protected pattern {protected} (MOCK_ORG_PROTECTED_HOSTS); refusing to write")
    client = rs.mongo_client(uri)
    db = client[db_name]
    if delegate_coll and colls.get(delegate_coll):
        master_agent_check(client, master_db, colls, delegate_coll, node_agents)
        stamp_agents_now(colls, delegate_coll, int(time.time()) + ahead_s)
    mkey = f"renter-manifest:{org['orgId']}"
    prev = (db[MANIFEST_COLL].find_one({"_id": mkey}) or {}).get("collections", {})

    # pre-flight: nothing is written until every collection passes
    plan = []
    for coll in list(colls) + [c for c in prev if c not in colls]:      # a collection switched off is cleaned too
        docs = colls.get(coll, [])
        keys = [id_key(d) for d in docs]
        kset = set(keys)
        stale = [k for k in prev.get(coll, []) if k not in kset]
        foreign = []
        if modes.get(coll) == "replace":
            mine = kset | set(prev.get(coll, []))
            foreign = [k for k in (key_of(d["_id"]) for d in db[coll].find({}, {"_id": 1})) if k not in mine]
            if foreign and not adopt:
                raise RenterError(f"{db_name}.{coll} holds {len(foreign)} document(s) this tool did not write; the UI reads a "
                                  f"single document there. Check it is safe, then re-run with --adopt (they are backed up and removed)")
        existing = sum(db[coll].count_documents({"_id": {"$in": [from_key(k) for k in part]}}) for part in chunks(keys)) if keys else 0
        plan.append((coll, docs, keys, stale, foreign, existing))

    print(f"\n  host {host}  database {db_name}  org {org['orgName']} ({org['orgId']})")
    print(f"  {'COLLECTION':48} {'MODE':8} {'WRITE':>7} {'REPLACE':>8} {'REMOVE':>7}")
    for coll, docs, keys, stale, foreign, existing in plan:
        print(f"  {coll:48} {modes.get(coll, 'cleanup'):8} {len(keys):>7} {existing:>8} {len(stale) + len(foreign):>7}")
    if not yes:
        if input(f"\n  type the database name ({db_name}) to write: ").strip() != db_name:
            raise RenterError("aborted; nothing written")

    bdir = Path(backup_dir) / re.sub(r"[^A-Za-z0-9._-]", "_", db_name) / time.strftime("%Y%m%d-%H%M%S")
    manifest = {}
    for coll, docs, keys, stale, foreign, existing in plan:
        c = db[coll]
        gone = stale + foreign
        old = []
        for part in chunks([from_key(k) for k in keys + gone]):
            old += list(c.find({"_id": {"$in": part}}))
        saved = None
        if old:
            bdir.mkdir(parents=True, exist_ok=True)
            saved = bdir / f"{coll}.ndjson"
            fd = os.open(saved, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for d in old:
                    fh.write(json_util.dumps(d) + "\n")
        deleted = 0
        for part in chunks([from_key(k) for k in gone]):
            deleted += c.delete_many({"_id": {"$in": part}}).deleted_count
        for part in chunks(docs):
            bson_docs = [json_util.loads(json.dumps(d)) for d in part]
            c.bulk_write([ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in bson_docs], ordered=False)
        found = sum(c.count_documents({"_id": {"$in": [from_key(k) for k in part]}}) for part in chunks(keys)) if keys else 0
        if found != len(keys):
            raise RenterError(f"{coll}: wrote {len(keys)} documents but found {found}" + (f"; backup in {saved}" if saved else ""))
        if modes.get(coll) == "replace" and c.count_documents({}) != len(keys):
            raise RenterError(f"{coll}: expected only this org's {len(keys)} document(s), found {c.count_documents({})}")
        if keys:
            manifest[coll] = keys
        print(f"  {coll:48} wrote {len(keys):>6}, removed {deleted:>5}" + (f"  (backup {saved})" if saved else ""))
    db[MANIFEST_COLL].replace_one({"_id": mkey}, {
        "_id": mkey, "tool": "populate_renter.py", "orgType": "RENTER", **meta,
        "uploadedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "collections": manifest}, upsert=True)
    print(f"  done; manifest {db_name}.{MANIFEST_COLL} {{_id: {mkey!r}}}")
    if delegate_coll and colls.get(delegate_coll):
        if ahead_s:
            until = time.strftime("%Y-%m-%d", time.gmtime(time.time() + ahead_s))
            print(f"  always connected: delegate sync times set to {until} - CONNECTED / healthy until then, nothing "
                  f"needs to keep running")
        elif beat:
            heartbeat(db, delegate_coll, [raw_id(d) for d in colls[delegate_coll]], beat)
        else:
            print("  note: delegates show CONNECTED for 5 minutes after this upload; add --heartbeat to keep them "
                  "connected while you test")


# ---------------------------------------------------------------- export (no connection from this machine)
MONGOSH_JS = r"""// Generated by populate_renter.py - loads mock RENTER org __ORG_NAME__ into database __DB__.
// Run with mongosh where the database is reachable:   mongosh "<connection string>" __FILE__
// or paste/load it in a mongosh session:              load("__FILE__")
// Upserts this org's documents by _id, removes only documents an earlier run of this tool wrote, never touches others.
(function () {
  const DB_NAME = __DB_JSON__;
  const ORG_ID = __ORG_ID_JSON__;
  const MKEY = "renter-manifest:" + ORG_ID;
  const MODES = __MODES_JSON__;
  const META = __META_JSON__;
  const DATA = EJSON.parse(__DATA_STRING__);
  const target = db.getSiblingDB(DB_NAME);
  const keyOf = (id) => (typeof id === "string" ? "s:" : "o:") + id.toString();
  const fromKey = (k) => (k.startsWith("o:") ? new ObjectId(k.slice(2)) : k.slice(2));
  const chunks = (xs, n) => { const out = []; for (let i = 0; i < xs.length; i += n) out.push(xs.slice(i, i + n)); return out; };

  const prevDoc = target.getCollection("_mock_org_tool").findOne({ _id: MKEY });
  const prev = (prevDoc && prevDoc.collections) || {};
  const names = Object.keys(DATA).concat(Object.keys(prev).filter((c) => !(c in DATA)));

  // pre-flight: nothing is written if a static collection holds documents this tool did not write
  const plan = names.map((coll) => {
    const docs = DATA[coll] || [];
    const keys = docs.map((d) => keyOf(d._id));
    const kset = new Set(keys);
    const stale = (prev[coll] || []).filter((k) => !kset.has(k));
    if (MODES[coll] === "replace") {
      const mine = new Set(keys.concat(prev[coll] || []));
      const foreign = target.getCollection(coll).find({}, { _id: 1 }).toArray().filter((d) => !mine.has(keyOf(d._id)));
      if (foreign.length) throw new Error(DB_NAME + "." + coll + " holds " + foreign.length + " document(s) this tool did not write; stopping, nothing written");
    }
    return { coll, docs, keys, stale };
  });

  print("database " + DB_NAME + "  org " + META.orgName + " (" + ORG_ID + ")");
  const manifest = {};
  for (const p of plan) {
    const c = target.getCollection(p.coll);
    let removed = 0;
    for (const part of chunks(p.stale.map(fromKey), 1000)) removed += c.deleteMany({ _id: { $in: part } }).deletedCount;
    for (const part of chunks(p.docs, 1000)) {
      c.bulkWrite(part.map((d) => ({ replaceOne: { filter: { _id: d._id }, replacement: d, upsert: true } })), { ordered: false });
    }
    let found = 0;
    for (const part of chunks(p.keys.map(fromKey), 1000)) found += c.countDocuments({ _id: { $in: part } });
    if (found !== p.keys.length) throw new Error(p.coll + ": wrote " + p.keys.length + " documents but found " + found);
    if (p.keys.length) manifest[p.coll] = p.keys;
    print("  " + p.coll.padEnd(48) + " wrote " + String(p.keys.length).padStart(6) + ", removed " + String(removed).padStart(5));
  }
  target.getCollection("_mock_org_tool").replaceOne({ _id: MKEY }, Object.assign({ _id: MKEY, tool: "populate_renter.py (mongosh)",
    orgType: "RENTER", uploadedAt: new Date().toISOString(), collections: manifest }, META), { upsert: true });
  print("done; manifest " + DB_NAME + "._mock_org_tool {_id: '" + MKEY + "'}");
})();
"""


def export_bundle(work, colls, modes, db_name, org, meta):
    """<work>/import/: one mongosh script (all documents inside) + one Extended-JSON array per collection + IMPORT.md."""
    out = work / "import"
    if out.exists():
        for p in out.iterdir():
            p.unlink()
    out.mkdir(exist_ok=True)
    js_name = f"load_{db_name}.js"
    js = MONGOSH_JS
    for k, v in (("__DB_JSON__", json.dumps(db_name)), ("__ORG_ID_JSON__", json.dumps(org["orgId"])),
                 ("__MODES_JSON__", json.dumps({c: modes[c] for c in colls})), ("__META_JSON__", json.dumps(meta)),
                 ("__DATA_STRING__", json.dumps(json.dumps(colls, ensure_ascii=False), ensure_ascii=False)),
                 ("__ORG_NAME__", org["orgName"]), ("__DB__", db_name), ("__FILE__", js_name)):
        js = js.replace(k, v)
    (out / js_name).write_text(js, encoding="utf-8")
    for coll, docs in colls.items():
        rs.save_json(out / f"{coll}.json", docs)
    lines = [f"# Import bundle - {org['orgName']} -> {db_name}", "",
             f"orgId `{org['orgId']}`, snapshot {meta['snapshot']}, seed {meta['seed']}.", "",
             "## Option A - mongosh (recommended: upserts, removes only this tool's old documents, verifies counts)", "",
             "```", f'mongosh "<connection string>" {js_name}', "```", "",
             f'Already inside a mongosh session on that server: `load("{js_name}")`.', "",
             "## Option B - mongoimport, one command per collection (upsert by _id)", "", "```"]
    lines += [f'mongoimport --uri "<connection string>" --db {db_name} --collection {c} --jsonArray --mode upsert --file {c}.json'
              for c in colls]
    lines += ["```", "", "Options B and C add / replace documents but do not remove documents of an earlier, different build "
              "(e.g. a workload you removed); use option A for re-runs.", "", "## Option C - MongoDB Compass", "",
              f"Open database `{db_name}`, for each collection below: Add Data -> Import JSON -> pick `<collection>.json`.", "",
              "| Collection | Documents |", "|---|---:|"]
    lines += [f"| {c} | {len(d)} |" for c, d in colls.items()]
    (out / "IMPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out, js_name


# ---------------------------------------------------------------- .env: kubectl port-forward + credentials from commands
ENV_KEYS = {"MONGO_URI", "MONGO_URI_CMD", "MONGO_USER", "MONGO_USER_CMD", "MONGO_PASSWORD", "MONGO_PASSWORD_CMD", "MONGO_HOST",
            "MONGO_AUTH_SOURCE", "MONGO_OPTIONS", "MONGO_CMD_BASE64", "KUBE_PORT_FORWARD_CMD", "KUBE_PORT_FORWARD_WAIT_S"}


def read_env_file(path):
    """KEY=VALUE lines; '#' comments, optional quotes and 'export '. Values still holding '<placeholder>' count as unset."""
    vals = {}
    for n, ln in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            raise RenterError(f"{path}:{n}: expected KEY=VALUE")
        k, v = s.split("=", 1)
        k, v = re.sub(r"^export\s+", "", k.strip()), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'" and v[0] not in v[1:-1]:   # "value" - not "a" b "c"
            v = v[1:-1]
        if k not in ENV_KEYS:
            raise RenterError(f"{path}:{n}: unknown key {k!r}; allowed: {sorted(ENV_KEYS)}")
        if v and "<" not in v:
            vals[k] = v
    return vals


def cmd_args(cmd):
    # Windows: hand the command line to CreateProcess as-is (double quotes work, backslashes in paths survive)
    return cmd if os.name == "nt" else shlex.split(cmd)


def run_secret_cmd(env, key):
    """Run the command in env[key]; return its trimmed output (base64-decoded if MONGO_CMD_BASE64=true). Never printed."""
    try:
        p = subprocess.run(cmd_args(env[key]), capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise RenterError(f"{key}: program not found ({env[key].split()[0]}); is kubectl installed and on PATH?")
    except subprocess.TimeoutExpired:
        raise RenterError(f"{key}: command did not finish within 60 s")
    if p.returncode:
        raise RenterError(f"{key} failed (exit {p.returncode}): {(p.stderr or p.stdout).strip()[:400]}")
    out = p.stdout.strip()
    if env.get("MONGO_CMD_BASE64", "").lower() in ("1", "true", "yes"):
        try:
            out = base64.b64decode(out, validate=True).decode().strip()
        except ValueError:
            raise RenterError(f"{key}: output is not base64; remove MONGO_CMD_BASE64 or use a base64decode go-template")
    if not out:
        raise RenterError(f"{key}: the command printed nothing")
    return out


def uri_from_env(env):
    """Full URI (MONGO_URI / MONGO_URI_CMD) or user + password (+ host, authSource, options); '' when nothing is set."""
    if env.get("MONGO_URI"):
        return env["MONGO_URI"]
    if env.get("MONGO_URI_CMD"):
        return run_secret_cmd(env, "MONGO_URI_CMD")
    user = env.get("MONGO_USER") or (run_secret_cmd(env, "MONGO_USER_CMD") if env.get("MONGO_USER_CMD") else "")
    pw = env.get("MONGO_PASSWORD") or (run_secret_cmd(env, "MONGO_PASSWORD_CMD") if env.get("MONGO_PASSWORD_CMD") else "")
    if not user and not pw:
        if env.get("KUBE_PORT_FORWARD_CMD"):                     # port forward only: try without a login
            opts = env.get("MONGO_OPTIONS", "directConnection=true")
            print("  no login in .env - connecting through the port forward without a user")
            return f"mongodb://{env.get('MONGO_HOST', 'localhost:27017')}/" + (f"?{opts}" if opts else "")
        return ""
    if not (user and pw):
        raise RenterError(".env gives only one of user / password; set both (MONGO_USER[_CMD] and MONGO_PASSWORD[_CMD])")
    host = env.get("MONGO_HOST", "localhost:27017")
    opts = env.get("MONGO_OPTIONS", "directConnection=true")
    return f"mongodb://{quote_plus(user)}:{quote_plus(pw)}@{host}/?authSource={env.get('MONGO_AUTH_SOURCE', 'admin')}" + \
           (f"&{opts}" if opts else "")


def port_open(host, port):
    with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=1):
        return True
    return False


@contextlib.contextmanager
def port_forward(env, uri):
    """Start KUBE_PORT_FORWARD_CMD in the background for the duration of the upload (skipped if the port is already open)."""
    cmd = env.get("KUBE_PORT_FORWARD_CMD")
    if not cmd or not uri.startswith("mongodb://"):                 # mongodb+srv (Atlas) and test URIs need no forward
        yield
        return
    hostport = rs.host_of(uri).split(",")[0].rsplit("@", 1)[-1]
    host, _, port = hostport.rpartition(":")
    host, port = host or "localhost", int(port) if port.isdigit() else 27017
    if host.lower() not in ("localhost", "127.0.0.1", "::1", "[::1]"):
        print(f"  URI points at {rs.host_of(uri)} (not localhost) - port forward not needed, not started")
        yield
        return
    if port_open(host, port):
        print(f"  port {host}:{port} already open - using the running port forward")
        yield
        return
    log = tempfile.TemporaryFile(mode="w+")
    try:
        proc = subprocess.Popen(cmd_args(cmd), stdout=log, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        raise RenterError(f"KUBE_PORT_FORWARD_CMD: program not found ({cmd.split()[0]}); is kubectl on PATH?")
    try:
        deadline = time.time() + float(env.get("KUBE_PORT_FORWARD_WAIT_S", 30))
        while not port_open(host, port):
            if proc.poll() is not None:
                log.seek(0)
                raise RenterError(f"port forward stopped (exit {proc.returncode}): {log.read().strip()[:400]}")
            if time.time() > deadline:
                raise RenterError(f"port forward did not open {host}:{port} within {env.get('KUBE_PORT_FORWARD_WAIT_S', 30)} s; "
                                  f"check the local port in KUBE_PORT_FORWARD_CMD matches MONGO_HOST")
            time.sleep(0.5)
        print(f"  port forward running on {host}:{port}")
        yield
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()
        print("  port forward stopped")


# ---------------------------------------------------------------- main
TEMPLATE = {
    "_comment": "RENTER org input for populate_renter.py. Put the real orgId of the new org and its database name here.",
    "org": {"orgName": "acme-labs.example.com", "orgId": "<ORG-ID-OF-THE-NEW-ORG>", "region": "us-east-1"},
    "mongo": {"database": "<DATABASE-OF-THE-NEW-ORG>"},
    "seed": 11,
    "snapshot": "now",
    "gpu": "H200",
    "gpuMetrics": {"temperatureC": {"idle": [28, 38], "busy": [40, 82]}, "memoryUsedPct": {"busy": [45, 90]}},
    "workloadsPerCluster": {"TRAINING_JOB": [0, 3], "INFERENCE_SERVICE": [5, 8]},
    "clusters": [{"name": "helios", "kind": "slurm", "nodes": 8, "gpu": "B300",
                  "workloads": {"TRAINING_JOB": [2, 3], "INFERENCE_SERVICE": [2, 4]}},
                 {"name": "atlas", "kind": "k8s", "nodes": [5, 7], "gpu": "GB200"},
                 {"name": "orion", "kind": "k8s", "nodes": 6}],
    "bandPct": {"idle": 20, "under": 30, "normal": 40, "over": 10},
    "storage": {"fleetTargetPct": 46.0, "fleetRangePct": [42.0, 50.0]},
}


def resolve_uri(arg, mongo):
    for src, v in (("--uri", arg), ("MONGO_URI", os.environ.get("MONGO_URI")), ("input mongo.uri", mongo.get("uri"))):
        v = (v or "").strip()
        if v and "<" not in v:
            if src == "input mongo.uri":
                print("  note: the connection string comes from the input file; prefer MONGO_URI so credentials stay out of files")
            return v
    return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="renter org JSON file")
    ap.add_argument("--template", action="store_true", help="print an input template and exit")
    ap.add_argument("--dry-run", action="store_true", help="build and validate, write JSON copies locally, nothing to MongoDB")
    ap.add_argument("--export", action="store_true",
                    help="no connection from here: write an import bundle (mongosh script + JSON per collection) to orgs/renter/<slug>/import/")
    ap.add_argument("--uri", help="MongoDB connection string (default: MONGO_URI env, then mongo.uri in the input)")
    ap.add_argument("--database", help="database of the org (default: mongo.database in the input)")
    ap.add_argument("--yes", action="store_true", help="do not ask before writing")
    ap.add_argument("--adopt", action="store_true", help="take over static_* collections that hold documents this tool did not write")
    ap.add_argument("--out", default=str(OUT_ROOT), help=f"output folder (default {OUT_ROOT})")
    ap.add_argument("--backup-dir", default=str(Path.home() / ".mock-org-backups"))
    ap.add_argument("--env", default=str(HERE / ".env"),
                    help="file with kubectl port-forward / credential commands (default: .env next to this script)")
    ap.add_argument("--heartbeat", nargs="?", type=int, const=60, default=0, metavar="SECONDS",
                    help="after the upload keep refreshing this org's delegate sync times (default every 60 s) until "
                         "Ctrl+C, so Settings > Integrations stays CONNECTED while you test (the backend needs < 5 min)")
    ap.add_argument("--always-connected", nargs="?", type=int, const=365, default=0, metavar="DAYS",
                    help="set the delegates' last-report time DAYS (default 365) in the future: the backend checks "
                         "now - lastSync <= 5 min, so Settings > Integrations stays CONNECTED with nothing running "
                         "(mock orgs only; the UI may show that future date as the last sync)")
    ap.add_argument("--node-agents", metavar="LIST", default="",
                    help="comma list of node-agent names to stamp on every delegate, e.g. 'dcgm_exporter,node_exporter' - "
                         "use when the upload cannot read caladrius_master.gpu_exporter_agent_health_configs")
    ap.add_argument("--master-db", default="caladrius_master",
                    help="master database read (read-only) for the agent names each resource type requires")
    ap.add_argument("--static", metavar="LIST",
                    help="which static_* collections to write this run: 'all', 'none', or a comma list such as "
                         "'static_resources,static_nodeagent_health'. Overrides the defaults and the input file. A static "
                         "collection left out is removed (only the document this tool wrote), so the backend falls back "
                         "to the original collections")
    a = ap.parse_args(argv)

    if a.template:
        print(json.dumps(TEMPLATE, indent=2))
        return 0
    if not a.input:
        ap.error("give the input JSON file (or --template)")

    try:
        req, mongo, names, target = load_input(a.input)
        if a.static is not None:
            static_kinds = [k for k in DEFAULT_COLLECTIONS if k.startswith("static_")]
            want = {"all": set(static_kinds), "none": set()}.get(a.static.strip().lower())
            if want is None:
                want = {x.strip() for x in a.static.split(",") if x.strip()}
                bad = sorted(want - set(static_kinds))
                if bad:
                    raise RenterError(f"--static: unknown {bad}; use all, none or any of {static_kinds}")
            for k in static_kinds:
                names[k] = k if k in want else None
            print(f"  static collections this run: {sorted(want) or 'none'}")
        org_given = "orgId" in req["org"]
        if a.export and not org_given:
            raise RenterError("--export needs the real org.orgId in the input (the UI shows only documents with that orgId)")
        if a.export and not (a.database or "<" not in str(mongo.get("database") or target.get("database") or "<")):
            raise RenterError("--export needs mongo.database in the input (or --database)")
        print(f"==== RENTER org {req['org'].get('orgName') or req['org'].get('name')}")
        try:
            work = rs.build_renter(req, Path(a.out))                           # engine build + verify_renter (raises on failure)
        except AssertionError as e:
            raise RenterError(f"renter engine internal check failed ({e or 'assertion'}); a fixed snapshot close to midnight "
                              f"or an extreme size can cause this - try snapshot 'now', another time of day or another seed")
        run = rs.load_json(work / "run.json")
        org = run["org"]
        snap_s = rs.epoch(dt.datetime.strptime(run["snapshot"], "%Y-%m-%dT%H:%M:%SZ"))
        requirement_check(work)
        kinds, want_types, want_docs = renter_docs(work, org, snap_s, run["seed"])
        check_docs(kinds, want_types, want_docs, org["orgId"])
        colls, modes = to_collections(kinds, names)
        out = work / "mongo"
        out.mkdir(exist_ok=True)
        for p in out.glob("*.json"):
            p.unlink()
        for coll, docs in colls.items():
            rs.save_json(out / f"{coll}.json", docs)
        print(f"\n  documents checked (counts, unique ids, relationships, project links, orgId) -> JSON copies in {out}")
        for coll, docs in colls.items():
            print(f"    {coll:48} {len(docs):>7}")
        if not org_given:
            print(f"\n  WARNING: no org.orgId in the input; generated {org['orgId']}. The UI shows these documents only for an "
                  f"org with this id - put the real orgId of the new org in the input.")

        db_name = a.database or mongo.get("database") or target.get("database")
        if db_name and "<" in db_name:
            db_name = None
        meta = {k: org.get(k) for k in ("orgId", "orgName", "orgSlug")}
        meta.update({"seed": run["seed"], "snapshot": run["snapshot"], "requestSha": run["requestSha"]})
        if a.export:
            if not org_given:
                raise RenterError("--export needs the real org.orgId in the input (the UI shows only documents with that orgId)")
            if not db_name:
                raise RenterError("--export needs mongo.database in the input (or --database)")
            if names.get("delegate_configs"):                                  # bundle loads later: stamp at build
                stamp_agents_now(colls, names["delegate_configs"], int(time.time()) + a.always_connected * 86400)
            bundle, js_name = export_bundle(work, colls, modes, db_name, org, meta)
            print(f"\n  import bundle -> {bundle}\n    {js_name}   (mongosh script, all documents inside)\n"
                  f"    IMPORT.md   (how to load it: mongosh, mongoimport or Compass)\n  nothing was written to MongoDB from here.")
            return 0
        if a.dry_run:
            print("\nDry run: nothing written.")
            return 0
        if not db_name:
            raise RenterError("set mongo.database in the input (or --database): the database of the new org")
        env = read_env_file(a.env) if Path(a.env).is_file() else {}
        uri = resolve_uri(a.uri, mongo)
        if not uri and env:
            print(f"  reading MongoDB access from {a.env}")
            uri = uri_from_env(env)                                             # may run kubectl; output never printed
        if not uri:
            print(f"\nNo MongoDB access (--uri, MONGO_URI, {a.env} or mongo.uri): nothing written.")
            return 0
        try:
            with port_forward(env, uri):
                upload(colls, modes, uri, db_name, org, meta, a.backup_dir, a.adopt, a.yes, a.master_db,
                       names.get("delegate_configs"), a.heartbeat, a.always_connected * 86400,
                       [x.strip() for x in a.node_agents.split(",") if x.strip()])
        except RenterError:
            raise
        except Exception as e:                                                  # noqa: BLE001 - turn driver errors into advice
            name, code = type(e).__name__, getattr(e, "code", None)
            if code in (13, 18) or "authentication" in str(e).lower() or "auth failed" in str(e).lower():
                raise RenterError(f"MongoDB refused the login ({name}: {str(e).split(', full error')[0]}). Put a user in the URI: "
                                  f"mongodb://USER:PASSWORD@host:port/?authSource=admin&directConnection=true "
                                  f"(for a database user use authSource={db_name}); the user needs readWrite on {db_name}")
            if name in ("ServerSelectionTimeoutError", "AutoReconnect", "ConnectionFailure", "NetworkTimeout"):
                raise RenterError(f"cannot reach MongoDB at {rs.host_of(uri)} ({name}); is the port forward running on that port?")
            raise
    except (RenterError, rs.RequestError) as e:
        print(f"\nERROR: {e}\nNothing was written to MongoDB unless listed above.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
