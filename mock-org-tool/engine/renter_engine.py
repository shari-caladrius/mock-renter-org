#!/usr/bin/env python3
"""Renter engine: builds the renter JSON set from a spec object (renter_spec.renter_spec).

Derived from renter-demo/generator/generate.py:
  * the spec is passed in (configure()), so any request can drive it;
  * dates are offsets from the snapshot day;
  * the trend sampler's count limits scale with the fleet size;
  * every cluster has its own GPU type (model, memory, GPUs per node, price) and metric ranges
    (temperature / power / memory used, separate for idle GPUs and GPUs running a workload);
  * generate(spec, out_dir) writes the JSON set.
"""
import calendar
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import health_catalog

REF = Path(__file__).resolve().parent / "reference"
OUT = None
spec = None

INFRA = {"k8s": "K8s", "slurm": "Slurm"}
RES_TYPE = {
    "k8s": {"label": "Kubernetes Cluster", "value": "kubernetes_cluster"},
    "slurm": {"label": "Slurm Cluster", "value": "slurm_cluster"},
}
SNAPSHOT = None  # naive UTC datetime, set by configure()
CONTAINER = {"INFERENCE_SERVICE": "dynamo-server", "TRAINING_JOB": "trainer"}
HEALTH_STATE = {"green": "Healthy", "degrading": "Degrading", "failing": "Failing"}
METRIC_SCORE = {"Healthy": 100, "Degrading": 50, "Failing": 0}
SCORE_BAND = {"Healthy": (93.0, 99.9), "Degrading": (55.0, 76.0), "Failing": (8.0, 35.0)}  # category weightedScore, seeded
GPU_OVERRIDE_KEYS = {"utilizationPct", "temperatureC", "memoryUsedMib", "powerDrawW"}
CFG = None
CFG_EFF = None


def configure(spec_obj, out_dir, config_doc=None):
    """Point the engine at a spec object (attributes as in renter-demo spec.py) and an output folder."""
    global spec, OUT, SNAPSHOT, CFG, CFG_EFF
    spec = spec_obj
    OUT = Path(out_dir)
    SNAPSHOT = datetime.strptime(spec.SNAPSHOT_UTC, "%Y-%m-%dT%H:%M:%SZ")
    CFG = config_doc or json.loads((REF / "config.json").read_text())
    CFG_EFF = CFG["data"]["gpuEfficiency"]


def day0():
    """Midnight of the snapshot day; absolute dates of the original generator are offsets from it."""
    return datetime(SNAPSHOT.year, SNAPSHOT.month, SNAPSHOT.day)


# ---------- helpers ----------
def digest(*parts):
    return hashlib.sha256("|".join([str(spec.SEED), *map(str, parts)]).encode()).hexdigest()


def oid(*parts):
    return digest(*parts)[:24]


def rng(*parts):
    return random.Random(int(digest(*parts)[:16], 16))


def r1(x):
    return round(x, 1)


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_epoch(dt):
    return calendar.timegm(dt.timetuple())


def clean(o):
    if isinstance(o, float) and o.is_integer():
        return int(o)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def write(name, obj, indent):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(clean(obj), indent=indent) + "\n")


def classify(util):
    if util <= CFG_EFF["idleThreshold"]:
        return "idle"
    if util < CFG_EFF["underUtilizedMax"]:
        return "under"
    if util < CFG_EFF["overUtilizedMin"]:
        return "normal"
    return "over"


def minutes_after(start, minutes_range, r):
    return start + timedelta(minutes=r.randrange(*minutes_range))


# ---------- fleet model ----------
def in_band(r, lo, hi):
    """Value inside [lo, hi], most often near the middle."""
    return r.triangular(lo, hi) if hi > lo else lo


def with_load(r, rng_lohi, util):
    """Grows with utilisation from the low to the high end of the range, a little noise, never outside it."""
    lo, hi = rng_lohi
    return clip(lo + (hi - lo) * util / 100.0 + r.gauss(0, (hi - lo) * 0.04), lo, hi)


def inside(x, lo, hi):
    """Round to 1 decimal without leaving [lo, hi]."""
    v = r1(x)
    return v if lo <= v <= hi else r1(clip(v, math.ceil(lo * 10) / 10, math.floor(hi * 10) / 10))


def sample_gpu(g, c):
    """Utilisation from the band; temperature, power and memory inside the cluster's ranges (idle or busy)."""
    r = rng("gpu", g["name"])
    R, total = c["ranges"], c["gpu"]["memoryMib"]
    state = "idle" if g["workload"] is None else "busy"
    t_r, p_r, m_r = R["temperatureC"][state], R["powerW"][state], R["memoryUsedPct"][state]
    if g["workload"] is None:
        util = 0.0
        temp, power, mem_pct = in_band(r, *t_r), in_band(r, *p_r), in_band(r, *m_r)
    else:
        mu, sd, lo, hi = spec.UTIL_MODEL[(g["kind"], g["band"])]
        util = r1(clip(r.gauss(mu, sd), lo, hi))
        temp, power = with_load(r, t_r, util), with_load(r, p_r, util)
        mem_pct = clip(r.gauss(g["mem_base"], (m_r[1] - m_r[0]) * 0.06), *m_r)
    mem = r1(mem_pct / 100 * total)
    while mem / total * 100 > m_r[1] and mem > 0:      # keep memory % inside the range after rounding
        mem = r1(mem - 0.1)
    while mem / total * 100 < m_r[0]:
        mem = r1(mem + 0.1)
    g.update(
        utilizationPct=r1(util),
        temperatureC=inside(temp, *t_r),
        memoryUsedMib=mem,
        memoryTotalMib=total,
        powerDrawW=inside(power, *p_r),
    )
    g.update(spec.OVERRIDES["gpu"].get(g["name"], {}))
    g["band"] = classify(g["utilizationPct"])


def spread_idle(cluster, node_gpus, total):
    """Pick `total` idle GPUs spread unevenly over the nodes, always the highest GPU indexes of a node."""
    r = rng("idle-spread", cluster)
    n = len(node_gpus)
    cap = min(5, len(node_gpus[0]))
    per_node = [total // n] * n
    for i in r.sample(range(n), total % n):
        per_node[i] += 1
    for _ in range(n * 2):  # random moves so the counts are not uniform
        a, b = r.randrange(n), r.randrange(n)
        if per_node[a] > 0 and per_node[b] < cap:
            per_node[a] -= 1
            per_node[b] += 1
    return {g["name"] for gs, k in zip(node_gpus, per_node) for g in gs[len(gs) - k:]}


def build_fleet():
    unknown = set(spec.OVERRIDES["gpu"])
    by_cluster = defaultdict(list)
    for w in spec.WORKLOADS:
        by_cluster[w["cluster"]].append(w)
    fleet = {"clusters": [], "nodes": [], "gpus": [], "workloads": []}
    for c in spec.CLUSTERS:
        infra = INFRA[c["kind"]]
        cluster_gpus, node_gpus = [], []
        for n in range(1, c["nodes"] + 1):
            node = spec.NODE_NAME.format(cluster=c["name"], rack=spec.RACK, n=n)
            gs = [{"name": spec.GPU_NAME.format(node=node, k=k), "parent": node, "cluster": c["name"],
                   "infra": infra, "gpuIndex": k, "model": c["gpu"]["model"], "workload": None}
                  for k in range(c["gpu"]["gpusPerNode"])]
            fleet["nodes"].append({"name": node, "cluster": c["name"], "index": n, "rack": spec.RACK, "gpus": gs})
            cluster_gpus += gs
            node_gpus.append(gs)
        idle_total = len(cluster_gpus) - sum(w["gpus"] for w in by_cluster[c["name"]])
        idle_names = spread_idle(c["name"], node_gpus, idle_total)
        free = [g for g in cluster_gpus if g["name"] not in idle_names]
        cursor = 0
        for w in by_cluster[c["name"]]:
            members = free[cursor:cursor + w["gpus"]]
            cursor += w["gpus"]
            b = w["buckets"]
            bands = ["under"] * b["under"] + ["normal"] * b["normal"] + ["over"] * b["over"]
            rng("bands", w["name"]).shuffle(bands)
            m_lo, m_hi = c["ranges"]["memoryUsedPct"]["busy"]          # one memory level per workload, mid-range
            mem_base = rng("membase", w["name"]).uniform(m_lo + (m_hi - m_lo) * 0.25, m_hi - (m_hi - m_lo) * 0.25)
            for g, band in zip(members, bands):
                g.update(workload=w["name"], kind=w["kind"], band=band, mem_base=mem_base)
            fleet["workloads"].append({**w, "infra": infra, "members": members})
        for g in cluster_gpus:
            sample_gpu(g, c)
            unknown.discard(g["name"])
        fleet["gpus"] += cluster_gpus
        fleet["clusters"].append({**c, "infra": infra, "gpus": cluster_gpus})
    if unknown:
        raise ValueError(f"OVERRIDES reference unknown GPUs: {sorted(unknown)}")
    for name, ov in spec.OVERRIDES["gpu"].items():
        if set(ov) - GPU_OVERRIDE_KEYS:
            raise ValueError(f"OVERRIDES gpu {name}: unsupported keys {sorted(set(ov) - GPU_OVERRIDE_KEYS)}")
    return fleet


def build_filesystems(fleet):
    st = spec.STORAGE
    fss = []
    for c in fleet["clusters"]:
        if c["kind"] == "slurm":
            fss.append({"name": spec.SHARED_NAME.format(cluster=c["name"]), "cluster": c["name"], "kind": "shared",
                        "node": None, "fsType": st["slurm_shared"][c["name"]]["fsType"],
                        "capacityGib": st["slurm_shared"][c["name"]]["capacityGib"]})
    for n in fleet["nodes"]:
        if n["cluster"] in {c["name"] for c in fleet["clusters"] if c["kind"] == "k8s"}:
            name = spec.SCRATCH_NAME.format(node=n["name"])
            r = rng("scratch", name)
            fss.append({"name": name, "cluster": n["cluster"], "kind": "scratch", "node": n["name"],
                        "fsType": "nvme", "capacityGib": r.choice(st["k8s_scratch_gib_choices"]),
                        "util": clip(r.gauss(46.0, 14.0), 18.0, 78.0)})
    scratch = [f for f in fss if f["kind"] == "scratch"]
    shared = [f for f in fss if f["kind"] == "shared"]
    cap_total = sum(f["capacityGib"] for f in fss)
    if scratch and not shared:     # K8s only: no shared filesystem to absorb the target, scale the scratch disks to it
        k = cap_total * st["fleet_util_pct_target"] / sum(f["capacityGib"] * f["util"] for f in scratch)
        for f in scratch:
            f["util"] = clip(f["util"] * k, 3.0, 97.0)
    for f in scratch:
        f["usedGib"] = r1(f["capacityGib"] * f["util"] / 100)
    need = cap_total * st["fleet_util_pct_target"] / 100 - sum(f["usedGib"] for f in scratch)
    offs = {f["name"]: rng("shared-offset", f["name"]).uniform(-3.0, 3.0) for f in shared}
    shared_cap = sum(f["capacityGib"] for f in shared)
    base = (need - sum(f["capacityGib"] * offs[f["name"]] / 100 for f in shared)) / shared_cap if shared_cap else 0.0
    if shared and not 0.02 < base < 0.95:
        raise ValueError(f"storage target {st['fleet_util_pct_target']}% cannot be reached with the shared filesystems "
                         f"(they would need {base * 100:.1f}% used); change the target or the shared capacity")
    for f in shared:
        f["usedGib"] = r1(f["capacityGib"] * (base * 100 + offs[f["name"]]) / 100)
    for f in fss:
        f["usedGib"] = spec.OVERRIDES["filesystem"].get(f["name"], {}).get("usedGib", f["usedGib"])
        f["freeGib"] = r1(f["capacityGib"] - f["usedGib"])
        f["utilPct"] = r1(f["usedGib"] / f["capacityGib"] * 100)
        f["mountPath"] = "/mnt/" + f["name"]
    missing = set(spec.OVERRIDES["filesystem"]) - {f["name"] for f in fss}
    if missing:
        raise ValueError(f"OVERRIDES reference unknown filesystems: {sorted(missing)}")
    fleet["filesystems"] = fss


def agg(gs):
    return {
        "avgUtilizationPct": r1(mean(g["utilizationPct"] for g in gs)),
        "avgTemperatureC": r1(mean(g["temperatureC"] for g in gs)),
        "avgMemoryUsedPct": r1(sum(g["memoryUsedMib"] for g in gs) / sum(g["memoryTotalMib"] for g in gs) * 100),
        "avgPowerDrawW": r1(mean(g["powerDrawW"] for g in gs)),
        "avgMemoryUsedMib": r1(mean(g["memoryUsedMib"] for g in gs)),
        "avgMemoryTotalMib": r1(mean(g["memoryTotalMib"] for g in gs)),
    }


# ---------- identities ----------
def build_ids(fleet):
    ids = {}
    for c in fleet["clusters"]:
        n = c["name"]
        r = rng("project-ts", n)
        created = minutes_after(day0() - timedelta(days=185), (0, 50 * 24 * 60), r)
        created = created.replace(second=0)
        ids[n] = {
            "projectId": oid("project", n),
            "resourceDocId": oid("resource-doc", n),
            "resourceId": digest("resourceId", n),
            "toolKey": digest("toolKey", n),
            "delegateKey": digest("delegateKey", n),
            "release": f"ds-{digest('release-a', n)[:11]}-{digest('release-b', n)[:24]}",
            "createdAt": to_epoch(created),
            "preparedAt": to_epoch(created) + r.choice([360, 420, 480, 540, 600]),
            "updatedAt": to_epoch(created) + r.choice(range(1020, 2161, 60)),
            "mongoCreatedAt": to_epoch(created) + r.choice(range(180, 601, 60)),
        }
    return ids


# ---------- API/DB documents ----------
def project_configs(fleet, ids):
    docs = []
    for c in fleet["clusters"]:
        i = ids[c["name"]]
        docs.append({
            "_id": {"$oid": i["projectId"]},
            "orgId": spec.ORG["orgId"],
            "orgName": spec.ORG["orgName"],
            "workSpaceId": "",
            "projectName": c["name"],
            "projectAlias": c["alias"],
            "referenceDb": f"{spec.ORG['orgName']}.{c['name']}.discover",
            "description": "",
            "status": "ACTIVE",
            "isDraft": False,
            "commandCenter": True,
            "commandCenterPreparedAt": {"$numberLong": str(i["preparedAt"])},
            "commandCenterIsUpdated": False,
            "createdAt": {"$numberLong": str(i["createdAt"])},
            "updatedAt": {"$numberLong": str(i["updatedAt"])},
            "createdBy": spec.ORG["createdBy"],
            "updatedBy": spec.ORG["createdBy"],
            "isDeleted": False,
            "region": {"hostingType": ""},
            "isComponentsSimulated": True,
            "simulatedComponentsCount": {"appComponents": 360, "infraEntities": 15},
            "useStaticGraph": True,
        })
    return docs


def k8s_install_command(release):
    ref = json.loads((REF / "resources.json").read_text())["data"][0]
    ref_release = [c["command"] for c in ref["installCommand"] if "helm upgrade" in c["command"]][0]
    ref_release = ref_release.split("--install ")[1].split(" ")[0]
    return [{**s, "command": s["command"].replace(ref_release, release)} for s in ref["installCommand"]]


def slurm_install_command(name, delegate_key):
    slug = digest("org-slug", name)[:11]
    script = (
        'curl -fsSL "https://lamintegration.blob.core.windows.net/installers/slurm/install-cluster.sh?<sas-token>" '
        "-o /tmp/caladrius-install-cluster.sh && \\\n"
        f"ORG_SLUG={slug} \\\nCLUSTER={name} \\\nENVIRONMENT=production \\\nREGION={spec.REGION} \\\n"
        f"DELEGATE_KEY={delegate_key} \\\nCOMPUTE_NODES=auto \\\nSSH_USER=<ssh-user> \\\n"
        "SSH_KEY=<path-to-ssh-private-key> \\\nENABLE_BMC=false \\\n"
        "LOKI_PUSH_URL=https://logs.caladrius.ai/loki/api/v1/push \\\n"
        "PROMETHEUS_REMOTE_WRITE_URL=https://metrics.caladrius.ai/api/v1/write \\\n"
        'PROMETHEUS_CA_CRT="<base64-ca-crt>" \\\nPROMETHEUS_TLS_CRT="<base64-tls-crt>" \\\n'
        'PROMETHEUS_TLS_KEY="<base64-tls-key>" \\\nALLOY_CA_CRT="<base64-ca-crt>" \\\n'
        'ALLOY_TLS_CRT="<base64-tls-crt>" \\\nALLOY_TLS_KEY="<base64-tls-key>" \\\n'
        'ACR_USERNAME=caladriusacr \\\nACR_PASSWORD="<acr-password>" \\\nbash /tmp/caladrius-install-cluster.sh'
    )
    return [
        {"title": "Verify Slurm Installation",
         "description": "Make sure the Slurm client tools are available on the head node.",
         "command": "sinfo --version"},
        {"title": "Verify Cluster Nodes",
         "description": "Confirm the compute nodes are visible to the scheduler.",
         "command": "sinfo -N -l"},
        {"title": "Install Caladrius Agent",
         "description": "Run the installer on the head node. It deploys the delegate and the node agents to the compute nodes.",
         "command": script},
    ]


def resources(fleet, ids):
    out = []
    for c in fleet["clusters"]:
        i = ids[c["name"]]
        cmd = k8s_install_command(i["release"]) if c["kind"] == "k8s" else slurm_install_command(c["name"], i["delegateKey"])
        out.append({
            "resourceId": i["resourceId"],
            "toolKey": i["toolKey"],
            "projectId": i["projectId"],
            "projectName": c["name"],
            "projectAlias": c["alias"],
            "name": c["name"],
            "alias": c["alias"],
            "type": RES_TYPE[c["kind"]],
            "provider": spec.PROVIDER,
            "region": spec.REGION,
            "delegateKey": i["delegateKey"],
            "installCommand": cmd,
            "agentStatus": "CONNECTED",
            "agenticLastSyncAt": spec.AGENT_SYNC_EPOCH_S,
            "clusterDiagnosticsEnabled": False,
        })
    return {"success": True, "message": "data retrieved successfully", "data": out}


def resources_mongo(fleet, ids):
    docs = []
    for c in fleet["clusters"]:
        i = ids[c["name"]]
        docs.append({
            "_id": {"$oid": i["resourceDocId"]},
            "orgId": spec.ORG["orgId"],
            "resourceId": i["resourceId"],
            "alias": c["alias"],
            "createdAt": {"$numberLong": str(i["mongoCreatedAt"])},
            "createdBy": spec.ORG["createdBy"],
            "details": "",
            "isDeleted": False,
            "name": c["name"],
            "projectId": {"$oid": i["projectId"]},
            "provider": spec.PROVIDER["value"].lower(),
            "region": spec.REGION,
            "toolKey": i["toolKey"],
            "type": RES_TYPE[c["kind"]]["value"],
            "updatedAt": {"$numberLong": str(i["mongoCreatedAt"])},
            "updatedBy": spec.ORG["createdBy"],
        })
    return docs


def nodeagent_health(fleet, ids):
    ms = spec.AGENT_SYNC_EPOCH_S * 1000 + 2427
    out = []
    for c in fleet["clusters"]:
        out.append({
            "delegateKey": ids[c["name"]]["delegateKey"],
            "referenceName": f"{c['name']}-agent",
            "resourceType": RES_TYPE[c["kind"]]["value"],
            "resourceConfigName": c["name"],
            "isHealthy": True,
            "agents": [
                {"name": "delegate", "displayName": "Delegate", "status": "healthy", "lastSyncAt": ms},
                {"name": "node_agents", "displayName": "Node Agents", "status": "healthy", "lastSyncAt": ms},
            ],
        })
    return {"success": True, "message": "exporter agent health retrieved successfully", "data": out}


# ---------- overview ----------
def workload_times(w, project_created_epoch):
    r = rng("wl-ts", w["name"])
    if w["kind"] == "INFERENCE_SERVICE":
        created = minutes_after(day0() - timedelta(days=129), (0, 107 * 24 * 60), r)
        updated = minutes_after(day0() - timedelta(days=3), (0, 3 * 24 * 60), r)
    else:
        created = minutes_after(day0() - timedelta(days=21), (0, 17 * 24 * 60), r)
        updated = minutes_after(day0() - timedelta(days=1), (0, 32 * 60), r)
    assert to_epoch(created) > project_created_epoch and updated < SNAPSHOT
    return iso(created.replace(second=0)), iso(updated.replace(second=0))


def overview(fleet, ids):
    gpus = fleet["gpus"]
    n = len(gpus)
    fs = fleet["filesystems"]
    cap = sum(f["capacityGib"] for f in fs)
    used = r1(sum(f["usedGib"] for f in fs))
    bands = {b: [g for g in gpus if g["band"] == b] for b in ("idle", "under", "normal", "over")}
    avg_util = r1(mean(g["utilizationPct"] for g in gpus))
    wl_entries = defaultdict(list)
    for w in fleet["workloads"]:
        created, updated = workload_times(w, ids[w["cluster"]]["createdAt"])
        wl_entries[w["kind"]].append({
            "canonicalName": w["name"],
            "name": w["name"],
            "subcategory": w["kind"],
            "category": "WORKLOAD",
            "cluster": w["cluster"],
            "gpu_devices": [{"canonicalName": g["name"], "name": g["name"], "type": "RUNS_ON"} for g in w["members"]],
            "createdAt": created,
            "updatedAt": updated,
            **agg(w["members"]),
        })
    ordered = wl_entries["INFERENCE_SERVICE"] + wl_entries["TRAINING_JOB"]
    by_type = {}                                     # one cost line per GPU type, first-seen order
    for c in fleet["clusters"]:
        t = by_type.setdefault(c["gpu"]["model"], {"instanceName": c["gpu"]["model"],
                                                   "priceCentsPerHour": c["gpu"]["priceCentsPerHour"], "gpuCount": 0})
        t["gpuCount"] += len(c["gpus"])
    for t in by_type.values():
        t["costPerHour"] = round(t["gpuCount"] * t["priceCentsPerHour"] / 100, 2)
    fleet_cost = round(sum(t["costPerHour"] for t in by_type.values()), 2)
    data = {
        "summary": {
            "totalGpuDevices": n,
            "totalInstances": len(fleet["nodes"]),
            "totalClusters": len(fleet["clusters"]),
            "standaloneInstances": 0,
            "avgUtilizationPct": avg_util,
            "avgTemperatureC": r1(mean(g["temperatureC"] for g in gpus)),
            "avgMemoryUsedPct": r1(sum(g["memoryUsedMib"] for g in gpus) / sum(g["memoryTotalMib"] for g in gpus) * 100),
            "avgPowerDrawW": r1(mean(g["powerDrawW"] for g in gpus)),
        },
        "clusters": [{"name": c["name"], "infra": c["infra"], "instanceCount": c["nodes"],
                      "totalGpuDevices": len(c["gpus"]), "avgUtilizationPct": r1(mean(g["utilizationPct"] for g in c["gpus"]))}
                     for c in fleet["clusters"]],
        "standaloneInstances": None,
        "gpuDevices": [{k: g[k] for k in ("name", "parent", "cluster", "infra", "gpuIndex", "model", "utilizationPct",
                                          "temperatureC", "memoryUsedMib", "memoryTotalMib", "powerDrawW")}
                       for g in gpus],
        "storage": {
            "totalCapacityGib": cap,
            "totalUsedGib": used,
            "totalFreeGib": r1(sum(f["freeGib"] for f in fs)),
            "utilPct": r1(used / cap * 100),
            "filesystems": [{k: f[k] for k in ("name", "cluster", "fsType", "mountPath", "capacityGib", "usedGib",
                                               "freeGib", "utilPct")} for f in fs],
        },
        "gpuEfficiency": {
            "idle": efficiency_list(bands["idle"]),
            "overUtilized": efficiency_list(bands["over"]),
            "underUtilized": efficiency_list(bands["under"]),
            "normal": efficiency_list(bands["normal"]),
            "total": n,
            "idleTotal": len(bands["idle"]),
            "underTotal": len(bands["under"]),
            "normalTotal": len(bands["normal"]),
            "overTotal": len(bands["over"]),
            "avgFleetUtilPct": avg_util,
            "durationHours": CFG_EFF["durationHours"],
            "dcgmDeviceCount": 0,
            "mfuDeviceCount": 0,
        },
        "costBreakdown": {
            "fleetCostPerHour": fleet_cost,
            "effectiveCostPerGpuHr": round(fleet_cost / n, 2),
            "wastedCapacityPerDay": round(fleet_cost * 24 * (1 - avg_util / 100), 2),
            "avgFleetUtil": avg_util,
            "byInstance": [{k: t[k] for k in ("instanceName", "priceCentsPerHour", "costPerHour", "gpuCount")}
                           for t in by_type.values()],
        },
        "workloadBreakdown": {"count": len(ordered), "byWorkload": [{"workloadName": w["name"]} for w in ordered]},
        "workload": {"INFERENCE_SERVICE": wl_entries["INFERENCE_SERVICE"], "TRAINING_JOB": wl_entries["TRAINING_JOB"]},
    }
    return {"success": True, "message": "data retrieved successfully", "data": data}


def efficiency_list(gs):
    return [{"name": g["name"], "cluster": g["cluster"], "infra": g["infra"], "avgUtilPct": g["utilizationPct"]} for g in gs]


# ---------- workload health ----------
def rollup_category(metrics):
    """Category health is the worst metric health; the score comes from SCORE_BAND. criticalOverride is always
    false (the UI misrenders it)."""
    states = {m["health"] for m in metrics}
    health = "Failing" if "Failing" in states else "Degrading" if "Degrading" in states else "Healthy"
    return health, None, False


def workload_health(fleet):
    computed = SNAPSHOT.strftime("%Y-%m-%dT%H:%M:%S") + ".625887414Z"
    groups = {"INFERENCE_SERVICE": [], "TRAINING_JOB": []}
    for w in fleet["workloads"]:
        if w["health"] not in HEALTH_STATE:
            raise NotImplementedError(f"workload {w['name']}: health '{w['health']}' not supported")
        issues = spec.HEALTH_ISSUES.get(w["name"], {}).get("metrics", {})
        known = {m["metricId"] for c in health_catalog.CATALOG[w["kind"]] for m in c["metrics"]}
        assert set(issues) <= known, (w["name"], set(issues) - known)
        a = agg(w["members"])
        ctx = {"mem_pct": a["avgMemoryUsedPct"], "max_temp": max(g["temperatureC"] for g in w["members"])}
        cats = []
        for cat in health_catalog.CATALOG[w["kind"]]:
            metrics = []
            for m in cat["metrics"]:
                r = rng("health", w["name"], m["metricId"])
                value, health, score = m["value"](r, ctx), "Healthy", m["score"]
                if m["metricId"] in issues:
                    health, v = issues[m["metricId"]]
                    value = r1(value * v[1]) if isinstance(v, tuple) else v
                    score = METRIC_SCORE[health] if m["isCritical"] or m["weight"] > 0 else -1
                metrics.append({"metricId": m["metricId"], "metricName": m["metricName"],
                                "currentValue": value, "unit": m["unit"], "health": health,
                                "isCritical": m["isCritical"], "weight": m["weight"], "score": score})
            health, _, _ = rollup_category(metrics)
            lo, hi = SCORE_BAND[health]
            score = r1(rng("health-score", w["name"], cat["categoryId"]).uniform(lo, hi))
            cats.append({"categoryId": cat["categoryId"], "categoryName": cat["categoryName"], "health": health,
                         "weightedScore": score, "effectiveWeight": cat["effectiveWeight"], "criticalOverride": False,
                         "metrics": metrics, "categoryDescription": cat["categoryDescription"]})
        # Overall health is the worst category health (Failing > Degrading > Healthy).
        worst = {c["health"] for c in cats}
        state = "Failing" if "Failing" in worst else "Degrading" if "Degrading" in worst else "Healthy"
        assert state == HEALTH_STATE[w["health"]], (w["name"], state, w["health"])
        groups[w["kind"]].append({
            "id": oid("workload", w["name"]), "workloadName": w["name"], "workloadType": w["kind"],
            "infraSource": w["infra"], "namespace": w["cluster"], "containerName": CONTAINER[w["kind"]],
            "overallHealth": state, "computedAt": computed, "cluster": "", "categories": cats,
        })

    def block(items):
        return {"total": len(items), "healthy": sum(i["overallHealth"] == "Healthy" for i in items),
                "degrading": sum(i["overallHealth"] == "Degrading" for i in items),
                "failing": sum(i["overallHealth"] == "Failing" for i in items), "workloads": items}

    return {"success": True, "message": "Workload health fetched successfully",
            "data": {"training": block(groups["TRAINING_JOB"]), "inference": block(groups["INFERENCE_SERVICE"])}}


# ---------- rate card ----------
def rate_card(fleet):
    """One document per cluster; price per node = GPU price x GPUs per node of that cluster, so it sums to the fleet cost."""
    docs = []
    for c in fleet["clusters"]:
        node_price = c["gpu"]["priceCentsPerHour"] * c["gpu"]["gpusPerNode"]
        r = rng("rate-card-ts", c["name"])
        updated = SNAPSHOT - timedelta(days=r.randrange(3, 21), minutes=r.randrange(0, 24 * 60))
        docs.append({
            "orgId": spec.ORG["orgId"],
            "clusterName": c["name"],
            "clusterType": "KUBERNETESCLUSTER" if c["kind"] == "k8s" else "SLURMCLUSTER",
            "instances": [{"entityName": n["name"], "priceCentsPerHour": node_price}
                          for n in fleet["nodes"] if n["cluster"] == c["name"]],
            "updatedAt": to_epoch(updated) * 1000 + r.randrange(0, 1000),
            "updatedBy": spec.ORG["createdBy"],
        })
    return {"success": True, "message": "Rate card fetched successfully", "data": docs}


# ---------- trend ----------
def trend_targets(n_days, today_avg):
    """Oldest-first daily averages that jump by at least `min_day_change_pct` between neighbours."""
    lo, hi = spec.TREND["history_fleet_util_pct_range"]
    gap = spec.TREND["min_day_change_pct"]
    r = rng("trend-targets")
    out, prev = [], today_avg
    for _ in range(n_days - 1):
        for _ in range(1000):
            t = r.uniform(lo, hi)
            if abs(t - prev) >= gap:
                break
        else:
            raise RuntimeError("cannot satisfy min_day_change_pct inside the history range")
        out.append(t)
        prev = t
    wlo, whi = spec.TREND["week_fleet_util_pct_range"]
    wgap = spec.TREND["week_min_day_change_pct"]
    w = spec.TREND["week_days"] - 1  # past days inside the 7d window; older days keep their targets
    rw = rng("trend-week-targets")
    for _ in range(1000):
        cand, prev = [], today_avg
        for _ in range(w):
            for _ in range(200):
                t = rw.uniform(wlo, whi)
                if abs(t - prev) >= wgap:
                    break
            cand.append(t)
            prev = t
        if abs(cand[-1] - out[w]) >= gap:
            break
    else:
        raise RuntimeError("cannot build the 7d window targets")
    out[:w] = cand
    return out[::-1]


def sample_day(r, mu, total, target, over_cap=150.0):
    """Pick bucket counts whose implied fleet average equals `target`."""
    idle0 = spec.BUCKET_TARGETS["idle"]
    jit = spec.TREND["idle_daily_jitter_pct"]
    s = total / 464.0  # limits below were tuned for 464 GPUs; scale them with the fleet
    for _ in range(5000):
        m = {"under": clip(r.gauss(mu["under"], 1.5), 5.0, 19.0), "normal": clip(r.gauss(mu["normal"], 3.0), 25.0, 75.0),
             "over": clip(r.gauss(mu["over"], 2.0), 82.0, 99.0)}
        idle = int(clip(round(r.gauss(idle0, idle0 * jit)), 30 * s, 160 * s))
        rest = total - idle
        # With u+n+o=rest and u*mu_u+n*mu_n+o*mu_o=target*total, n = a - k*o.
        a = (target * total - rest * m["under"]) / (m["normal"] - m["under"])
        k = (m["over"] - m["under"]) / (m["normal"] - m["under"])
        o_lo = max(4.0 * s, (60 * s - rest + a) / (k - 1), (a - 230 * s) / k)
        o_hi = min((a - 20 * s) / k, over_cap * s)
        if o_lo > o_hi:
            continue
        over = round(r.uniform(o_lo, o_hi))
        normal = round(a - k * over)
        under = total - idle - normal - over
        if min(under, normal, over) < 1:
            continue
        avg = r1((under * m["under"] + normal * m["normal"] + over * m["over"]) / total)
        if abs(avg - target) <= 0.25:
            return avg, idle, under, normal, over
    raise RuntimeError(f"no bucket counts reach a fleet average of {target:.1f}")


def trend(fleet):
    gpus = fleet["gpus"]
    total = len(gpus)
    bands = {b: [g["utilizationPct"] for g in gpus if g["band"] == b] for b in ("idle", "under", "normal", "over")}
    mu = {b: mean(bands[b]) for b in ("under", "normal", "over")}
    last = SNAPSHOT.date()
    n_days = spec.TREND["window_days"]
    today_avg = r1(mean(g["utilizationPct"] for g in gpus))
    targets = trend_targets(n_days, today_avg)
    days = []
    for i in range(n_days):
        d = last - timedelta(days=n_days - 1 - i)
        if d == last:
            row = (today_avg, len(bands["idle"]), len(bands["under"]), len(bands["normal"]), len(bands["over"]))
        else:
            in_week = 0 < n_days - 1 - i <= spec.TREND["week_days"] - 1
            row = sample_day(rng("trend", d.isoformat()), mu, total, targets[i], 200.0 if in_week else 150.0)
        days.append({"date": d.isoformat(), "avgFleetUtil": row[0], "idleCount": row[1], "underCount": row[2],
                     "normalCount": row[3], "overCount": row[4], "totalDevices": total})
    return {n: {"success": True, "message": "GPU efficiency trend fetched successfully",
                "data": {"days": days[-n:], "durationDays": n}} for n in (1, 7, 15, 30)}


# ---------- inventory ----------
def inventory(fleet, ids):
    fs_by_node = {f["node"]: f["name"] for f in fleet["filesystems"] if f["node"]}
    shared = {f["cluster"]: f["name"] for f in fleet["filesystems"] if f["kind"] == "shared"}
    return {
        "snapshot": spec.SNAPSHOT_UTC,
        "org": {"orgId": spec.ORG["orgId"], "orgName": spec.ORG["orgName"]},
        "totals": {"clusters": len(fleet["clusters"]), "nodes": len(fleet["nodes"]), "gpus": len(fleet["gpus"]),
                   "workloads": len(fleet["workloads"]), "storage": len(fleet["filesystems"])},
        "clusters": [{
            "name": c["name"], "alias": c["alias"], "kind": c["kind"], "infra": c["infra"],
            "projectId": ids[c["name"]]["projectId"], "resourceId": ids[c["name"]]["resourceId"],
            "resourceDocId": ids[c["name"]]["resourceDocId"], "toolKey": ids[c["name"]]["toolKey"],
            "delegateKey": ids[c["name"]]["delegateKey"], "nodeCount": c["nodes"], "gpuCount": len(c["gpus"]),
            "gpu": c["gpu"], "metricRanges": c["ranges"],
            "workloads": [w["name"] for w in fleet["workloads"] if w["cluster"] == c["name"]],
            "storage": [f["name"] for f in fleet["filesystems"] if f["cluster"] == c["name"]],
        } for c in fleet["clusters"]],
        "nodes": [{
            "name": n["name"], "cluster": n["cluster"], "rack": n["rack"], "index": n["index"],
            "gpus": [g["name"] for g in n["gpus"]],
            "storage": fs_by_node.get(n["name"], shared.get(n["cluster"])),
        } for n in fleet["nodes"]],
        "gpus": [{"name": g["name"], "node": g["parent"], "cluster": g["cluster"], "index": g["gpuIndex"],
                  "workload": g["workload"], "band": g["band"], "utilizationPct": g["utilizationPct"]}
                 for g in fleet["gpus"]],
        "workloads": [{"id": oid("workload", w["name"]), "name": w["name"], "kind": w["kind"], "cluster": w["cluster"],
                       "gpuCount": len(w["members"]), "gpus": [g["name"] for g in w["members"]]}
                      for w in fleet["workloads"]],
        "storage": [{"name": f["name"], "cluster": f["cluster"], "kind": f["kind"], "node": f["node"],
                     "fsType": f["fsType"], "capacityGib": f["capacityGib"], "usedGib": f["usedGib"],
                     "utilPct": f["utilPct"]} for f in fleet["filesystems"]],
    }


def generate(spec_obj, out_dir, config_doc=None):
    configure(spec_obj, out_dir, config_doc)
    fleet = build_fleet()
    build_filesystems(fleet)
    ids = build_ids(fleet)
    ov = overview(fleet, ids)
    write("config.json", CFG, 4)
    write("overview.json", ov, 4)
    write("project_configs.json", project_configs(fleet, ids), 2)
    write("resources.json", resources(fleet, ids), 4)
    write("resources-mongo.json", resources_mongo(fleet, ids), 2)
    write("nodeagent-health.json", nodeagent_health(fleet, ids), 4)
    write("workload-health.json", workload_health(fleet), 4)
    write("rate-card.json", rate_card(fleet), 4)
    for n, t in trend(fleet).items():
        write(f"gpu-efficiency-trend-{n}d.json", t, 4)
    write("inventory.json", inventory(fleet, ids), 2)
    d = ov["data"]
    print(f"gpus={d['summary']['totalGpuDevices']} fleetUtil={d['summary']['avgUtilizationPct']}% "
          f"storageUtil={d['storage']['utilPct']}% idle/under/normal/over="
          f"{d['gpuEfficiency']['idleTotal']}/{d['gpuEfficiency']['underTotal']}/"
          f"{d['gpuEfficiency']['normalTotal']}/{d['gpuEfficiency']['overTotal']}")
    print(f"wrote {len(list(OUT.glob('*.json')))} files to {OUT}")
    return fleet
