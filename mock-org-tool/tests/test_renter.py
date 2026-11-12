#!/usr/bin/env python3
"""Checks for populate_renter.py - nothing is written to a real database.

    python tests/test_renter.py

1. every request in requests/ builds and passes all checks (dry run, temp folder)
2. every GPU type in gpu_catalog.json builds with its default ranges
3. bad inputs are refused with a clear message
4. upload to an in-memory MongoDB (only if `pip install mongomock` is done; skipped otherwise):
   documents land, a re-run with a different GPU type removes the old GPUs, delegates stay CONNECTED
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import populate_renter as pr  # noqa: E402
import renter_spec as rs  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="renter-test-"))
ORG = {"orgName": "test-renter.example.com", "orgId": "11111111-2222-3333-4444-555555555555", "orgSlug": "testrenter"}
results = []


def main_quiet(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = pr.main(args)
    return rc, out.getvalue(), err.getvalue()


def build(req_or_path, extra=()):
    p = req_or_path
    if isinstance(req_or_path, dict):
        p = TMP / "in.json"
        p.write_text(json.dumps(req_or_path))
    return main_quiet([str(p), "--dry-run", "--out", str(TMP / "out"), "--env", str(TMP / "no.env"), *extra])


def step(name, fn):
    try:
        fn()
        results.append((name, None))
    except Exception as e:  # noqa: BLE001
        results.append((name, f"{type(e).__name__}: {e}"))


def ok(req_or_path):
    rc, out, err = build(req_or_path)
    assert rc == 0, (err or out)[-800:]
    assert "validation: 0 errors" in out and "every cluster matches its request" in out, out[-800:]


def refused(req, text):
    rc, out, err = build(req)
    assert rc == 2 and text in err, f"rc {rc}: {err.strip()[:300]}"


for f in sorted((ROOT / "requests").glob("*.json")):
    step(f"request {f.name}", lambda f=f: ok(f))
for t in sorted(k for k in rs.gpu_catalog()):
    step(f"GPU type {t}", lambda t=t: ok({"org": ORG, "seed": 3, "gpu": t, "clusters": [
        {"name": "alpha", "kind": "slurm", "nodes": [4, 6]}, {"name": "beta", "kind": "k8s", "nodes": 5}]}))
step("K8s-only fleet reaches the storage target", lambda: ok({"org": ORG, "clusters": [{"name": "alpha", "kind": "k8s", "nodes": 4}]}))
step("optional workloadHealth still works", lambda: ok({"org": ORG, "clusters": [{"name": "alpha", "kind": "k8s", "nodes": 6}],
                                                       "workloadHealth": {"failing": 1, "degrading": 1}}))

BASE = {"org": ORG, "clusters": [{"name": "alpha", "kind": "k8s", "nodes": 4}]}
for name, extra, text in [
        ("old fleet-wide workload counts", {"workloads": {"TRAINING_JOB": 2}}, "per cluster"),
        ("unknown GPU type", {"gpu": "X999"}, "not in gpu_catalog"),
        ("range min > max", {"gpuMetrics": {"temperatureC": [90, 40]}}, "min <= max"),
        ("memory above 100 %", {"gpuMetrics": {"memoryUsedPct": {"busy": [50, 120]}}}, "<= 100"),
        ("unknown metric", {"gpuMetrics": {"temp": [40, 90]}}, "unknown key"),
        ("more workloads than GPUs", {"clusters": [{"name": "alpha", "kind": "k8s", "nodes": 2,
                                                    "workloads": {"INFERENCE_SERVICE": 20}}]}, "workloads but only"),
        ("GPU model in cluster name", {"clusters": [{"name": "gb200-pool", "kind": "k8s", "nodes": 4}]}, "cluster name")]:
    step(f"refuse {name}", lambda e=extra, t=text: refused({**BASE, **e}, t))


def upload_round_trip():
    import mongomock

    def bulk(self, ops, ordered=True):                     # mongomock + pymongo 4: ReplaceOne attributes
        for op in ops:
            self.replace_one(op._filter, op._doc, upsert=op._upsert)
    mongomock.collection.Collection.bulk_write = bulk
    client = mongomock.MongoClient()
    rs.mongo_client = lambda uri: client
    req = {**rs.load_json(ROOT / "requests" / "renter-pxz7ru3sqvx.json"), "org": ORG, "mongo": {"database": "caladrius_test"}}
    common = ["--uri", "mongomock://t", "--yes", "--static", "none", "--always-connected",
              "--node-agents", "dcgm-exporter,node-exporter", "--backup-dir", str(TMP / "bak")]
    rc, out, err = build(req, common)                        # --dry-run first: nothing written
    assert rc == 0 and "caladrius_test" not in client.list_database_names()
    p = TMP / "up.json"
    p.write_text(json.dumps(req))
    rc, out, err = main_quiet([str(p), "--out", str(TMP / "out"), "--env", str(TMP / "no.env"), *common])
    assert rc == 0, err[-800:]
    db = client["caladrius_test"]
    assert db["caladrius_neo_cloud_entities"].count_documents({"type": "GPU_DEVICE"}) == 160
    snaps = list(db["caladrius_neo_cloud_workload_health_snapshots"].find())
    assert snaps and all(s["overall_health"] == "Healthy" for s in snaps), "workload health must be all Healthy"
    now = int(time.time())
    for d in db["delegate_configs"].find():
        assert d["agenticLastSyncAt"] > now + 86400, "always-connected: sync time in the future"
        assert {"dcgm-exporter", "node-exporter"} <= set(d["nodeAgentConfigs"])
    # switch atlas to GB200 (4 GPUs per node): the H200 GPUs of atlas must disappear, nothing else of the org duplicated
    req["clusters"][1]["gpu"] = "GB200"
    p.write_text(json.dumps(req))
    rc, out, err = main_quiet([str(p), "--out", str(TMP / "out"), "--env", str(TMP / "no.env"), *common])
    assert rc == 0, err[-800:]
    atlas = list(db["caladrius_neo_cloud_entities"].find({"type": "GPU_DEVICE", "cluster": "atlas"}))
    assert atlas and all(g["attributes"]["model"].startswith("NVIDIA GB200") for g in atlas)
    assert len(atlas) == 6 * 4 and not any(g["canonicalName"].endswith(("-gpu-4", "-gpu-7")) for g in atlas)


try:
    import mongomock  # noqa: F401
    step("upload round trip (mongomock)", upload_round_trip)
except ImportError:
    print("upload round trip skipped: pip install mongomock to run it")

shutil.rmtree(TMP, ignore_errors=True)
print("\n".join(f"{'PASS' if e is None else 'FAIL'}  {n}" + (f"\n      {e}" if e else "") for n, e in results))
failed = sum(e is not None for _, e in results)
print(f"\n{len(results) - failed} passed, {failed} failed")
sys.exit(1 if failed else 0)
