# Renter mock org

Builds one mock **RENTER** org from a JSON file and loads it into that org's MongoDB database, so the Caladrius UI
(GPU fleet, efficiency, workloads, Settings > Integrations, agent health) renders from it.

```
populate_renter.py        the command: build -> check -> upload (or --dry-run / --export)
renter_spec.py            reads the input, draws the ranges, verifies everything the engine wrote
gpu_catalog.json          GPU types (H100, H200, B200, B300, GB200, GB300, A100, L40S, MI300X) - edit or add here
engine/                   data generator (renter_engine.py, health_catalog.py, reference/ = key-shape references)
requests/                 input files
tests/test_renter.py      python tests/test_renter.py  (upload test runs when mongomock is installed)


orgs/renter/<slug>/       output of the last run: generated/, mongo/ (JSON per collection), SUMMARY.md
```

## Run

```
python populate_renter.py requests/renter-pxz7ru3sqvx.json --dry-run
python populate_renter.py requests/renter-pxz7ru3sqvx.json --static none --always-connected --node-agents dcgm-exporter,node-exporter,topology-collector,filesystem-probe-exporter,log-exporter,ksm,rdma-exporter,amd_dme_exporter
```

`--dry-run` builds and checks, writes nothing to MongoDB. Without it the tool connects through `.env` (or `MONGO_URI` /
`--uri`), shows what it will write, and asks you to type the database name (`--yes` skips the question).

| Option | What it does |
|---|---|
| `--static none` | no `static_*` collections; the backend reads the original collections |
| `--always-connected [DAYS]` | delegate sync time set DAYS (365) ahead, so Integrations stays CONNECTED |
| `--heartbeat [SECONDS]` | instead: keep refreshing the sync time until Ctrl+C |
| `--node-agents LIST` | agent names stamped on every delegate (when `caladrius_master` is not readable) |
| `--export` | no connection: writes a mongosh script + JSON per collection to `orgs/renter/<slug>/import/` |
| `--template` | prints a starting input |

## Input

| Key | Value |
|---|---|
| `org` | `orgName`, `orgId` (the real id of the new org), `orgSlug`, `region` |
| `mongo` | `{"database": "caladrius_<slug>"}` |
| `seed` | same seed + same input = same data; every range below is drawn from it |
| `snapshot` | `"now"` or `"YYYY-MM-DDTHH:MM:SSZ"` |
| `gpu` | fleet default GPU type: a key of `gpu_catalog.json` (`"H200"`, `"GB200"`, `"B300"`, ...) |
| `gpuMetrics` | fleet default metric ranges (below) |
| `workloadsPerCluster` | workload counts for clusters without their own; default `{"TRAINING_JOB": [0, 3], "INFERENCE_SERVICE": [5, 8]}` |
| `clusters` | list, see below; or `{"count": 6, "slurm": 2, "k8s": 4, "nodes": [7, 12]}` |
| `workloads` | optional `{"items": [{"name", "cluster", "kind", "gpus"}]}`: exact workloads instead of counts |
| `bandPct` | share of GPUs idle / under / normal / over, e.g. `{"idle": 20, "under": 30, "normal": 40, "over": 10}` |
| `storage` | `{"fleetTargetPct": 46.0, "fleetRangePct": [42.0, 50.0]}` |
| `utilization` | optional `{"fleetRangePct": [30, 40]}`: checks the fleet average lands inside |
| `overrides` | optional fixed values for one GPU / filesystem, e.g. a hot GPU (exempt from the ranges) |
| `workloadHealth` | optional; **absent = every workload Healthy** (the data pipeline computes the real health) |

### Clusters

```json
{ "name": "helios", "kind": "slurm", "nodes": 8, "gpu": "B300",
  "workloads": { "TRAINING_JOB": [1, 3], "INFERENCE_SERVICE": [2, 4] },
  "gpuMetrics": { "temperatureC": { "busy": [35, 70] } } }
```

- `kind`: `slurm` or `k8s`. `nodes`: a number or a range `[min, max]`.
- `workloads`: counts per kind, a number or a range; omitted = `workloadsPerCluster`. Both kinds can run in any cluster.
- `gpu`: omitted = the fleet `gpu`. GB200 / GB300 have 4 GPUs per node (one compute tray), the others 8.
- Cluster names must not contain GPU-model or orchestrator words (`h200`, `slurm`, ...).

### Metric ranges (`gpuMetrics`, fleet-wide or per cluster)

```json
"gpuMetrics": {
  "temperatureC":  { "idle": [28, 38],   "busy": [40, 83] },
  "powerW":        { "idle": [68, 94],   "busy": [98, 700] },
  "memoryUsedPct": { "idle": [1.5, 5.0], "busy": [45, 90] }
}
```

- `busy` = GPUs running a workload, `idle` = GPUs without one. `[min, max]` alone means `busy`.
- Busy temperature and power rise with utilisation from the low to the high end of the range; memory is set per workload.
- The 30 days of metric history stay inside the same ranges. The tool refuses to write if any value falls outside.
- Defaults come from the GPU type: power from its idle power and TDP (`maxPowerW`); temperature from cooling
  (air: idle 28-38, busy 40-83 °C; liquid: idle 25-34, busy 32-68 °C); memory idle 1.5-5 %, busy 45-90 %.
- Order: GPU-type default, then fleet `gpuMetrics`, then the cluster's `gpuMetrics`.

### GPU types (`gpu_catalog.json`)

| Type | Model | GPUs/node | Memory | Price /GPU-hr | Power idle / TDP | Cooling |
|---|---|---:|---:|---:|---|---|
| A100 | NVIDIA A100 80GB HBM2e | 8 | 80 GB | $1.29 | 55 / 400 W | air |
| L40S | NVIDIA L40S 48GB GDDR6 | 8 | 48 GB | $0.99 | 35 / 350 W | air |
| H100 | NVIDIA H100 80GB HBM3 | 8 | 80 GB | $2.49 | 70 / 700 W | air |
| H200 | NVIDIA H200 141GB HBM3e | 8 | 141 GB | $3.99 | 75 / 700 W | air |
| B200 | NVIDIA B200 180GB HBM3e | 8 | 180 GB | $5.49 | 120 / 1000 W | air |
| B300 | NVIDIA B300 288GB HBM3e | 8 | 288 GB | $9.47 | 140 / 1100 W | liquid |
| GB200 | NVIDIA GB200 186GB HBM3e | 4 | 186 GB | $6.99 | 130 / 1200 W | liquid |
| GB300 | NVIDIA GB300 288GB HBM3e | 4 | 288 GB | $9.99 | 150 / 1400 W | liquid |
| MI300X | AMD Instinct MI300X 192GB HBM3 | 8 | 192 GB | $2.99 | 90 / 750 W | air |

A cluster can override single fields: `"gpu": {"type": "H200", "priceCentsPerHour": 350}`.

## Example inputs (`requests/`)

| File | Shows |
|---|---|
| `renter-pxz7ru3sqvx.json` | your org: H200, workload counts per cluster from ranges, explicit metric ranges |
| `renter-pxz7ru3sqvx-mixed-gpus.json` | same org, mixed fleet: B300 Slurm, GB200 K8s (node range), H200 K8s |
| `renter-pxz7ru3sqvx-rich.json` | same org, exact named workloads, aliases, one hot GPU |
| `renter-template.json` | starting point for a new org |
| `renter-example-minimal.json` | smallest input (dry run only) |

All three `pxz7ru3sqvx` files write to the same database; each run replaces the previous run's documents (the manifest in
`_mock_org_tool` removes what is no longer generated, e.g. the H200 GPUs when a cluster switches to GB200).

## What the checks cover

Before anything is written: node / GPU / workload counts per cluster against the request and its ranges, GPU model and
memory per cluster, every temperature / power / memory value (today and 29 days of history) inside its range, efficiency
bands, cost (rate card = sum of each cluster's price x GPUs), storage target, relationships between entities, Integrations
joins, unique ids, one orgId.
