# Removing the static collections, one screen at a time

Goal: the renter org works with **no `static_*` collection**; every screen reads the original collections.

Each step removes **one more** static collection with `--static` (the list = the static collections that stay).
The tool deletes only the static document it wrote earlier (backed up first), so the backend falls back to the
original collections for that screen. If a screen breaks, run the previous step's command again to restore it.

Run everything from `C:\Users\Ctr_Sriraam\Downloads\mock-org-tool\mock-org-tool`; input `requests/renter-pxz7ru3sqvx.json`.
`...` = `python populate_renter.py requests/renter-pxz7ru3sqvx.json`

Note: the numbers in "Expected result" below were measured with the earlier fixed workload counts (2 training + 8 inference);
workload counts are now drawn per cluster from ranges, so compare against the summary the run prints.

**Add `--always-connected` once `static_resources` is removed** (step 5 on): the backend shows an agent CONNECTED only if
`now - lastSync <= 5 minutes` (`AGENT_SYNC_STALENESS_THRESHOLD_SEC = 300`). `--always-connected` sets the delegates' last
report one year ahead, so every check passes and nothing has to keep running (`--always-connected 30` = 30 days).
Alternative: `--heartbeat` refreshes the time every 60 s while its window stays open (Disconnected 5 min after it stops).
In the commands below, `--heartbeat` can be replaced by `--always-connected`.

| # | Screen / endpoint | Static removed | Backend reads instead | Command | Result |
|---|---|---|---|---|---|
| 0 | Baseline | none | - | `... --static all` | |
| 1 | Rate card `/neo-cloud/rate-card` | `static_rate_card` | `caladrius_neo_cloud_rate_cards` | `... --static static_config,static_gpu_efficiency_trend,static_workload_health,static_neocloud_overview,static_resources,static_nodeagent_health` | |
| 2 | Efficiency trend `/neo-cloud/gpu-efficiency-trend` | `static_gpu_efficiency_trend` | `caladrius_neo_cloud_gpu_metrics` (30 days of history) | `... --static static_config,static_workload_health,static_neocloud_overview,static_resources,static_nodeagent_health` | |
| 3 | Workload health `/neo-cloud/workloads/health` | `static_workload_health` | entities (TRAINING_JOB / INFERENCE_SERVICE) + `..._workload_health_snapshots` | `... --static static_config,static_neocloud_overview,static_resources,static_nodeagent_health` | |
| 4 | Overview `/neo-cloud/overview` | `static_neocloud_overview` | `caladrius_neo_cloud_entities` + `caladrius_neo_cloud_gpu_metrics` + rate cards | `... --static static_config,static_resources,static_nodeagent_health` | |
| 5 | Integrations `/self-managed/resources` | `static_resources` | `self_managed_resources` + `delegate_configs` + `caladrius_monitored_entities` (+ master templates) | `... --static static_config,static_nodeagent_health --heartbeat` | |
| 6 | Agent health `/exporter-agent-health` | `static_nodeagent_health` | `delegate_configs.nodeAgentConfigs` (+ master agent config) | `... --static static_config --heartbeat` | |
| 7 | Thresholds | `static_config` | `caladrius_neo_cloud_obs_config` (org, then master, then defaults 0 / 20 / 81) | `... --static none --heartbeat` | |

## Expected result per step (backend logic simulated on the generated data)

| # | Expected |
|---|---|
| 1 | Fleet cost $638.40/hr (backend: each node in the rate cards counts once x $31.92) |
| 2 | 7 daily points around 35% fleet utilisation |
| 3 | **Only if Grafana is configured for the org** (see below): 10 workloads, 1 failing, 2 degrading, 7 healthy. Otherwise every workload shows "NoData". |
| 4 | GPU clusters 3: K8s atlas (6 instances, 48 GPUs, 21.2%), K8s orion (6, 48, 25.4%), Slurm helios (8, 64, 53.7%), no Standalone box; 160 GPUs, 35.4%, storage 46% (204.2 / 444 TiB); efficiency 32 idle / 48 under / 64 normal / 16 over; workloads 2 training + 8 inference |
| 5 | Atlas, Helios, Orion CONNECTED with install commands (needs master `delegate_install_templates` for `slurm_cluster` and `kubernetes_cluster` - the upload warns if missing) |
| 6 | Each cluster: Delegate + Node Agents healthy (the upload reads the master agent list for each type and stamps every agent) |
| 7 | Bands from the master / default config (over >= 81) |

## Known backend behaviour (not fixable from the data)

1. **Workload health** is now computed by the data pipeline (reads Grafana, stores the result in MongoDB); Wrench reads that
   stored data. The tool writes every workload as **Healthy** in `caladrius_neo_cloud_workload_health_snapshots`. (In the
   older wrench-service version read earlier, `computeWorkloadsHealth` checked Grafana before reading snapshots, which showed
   NoData for mock orgs.)
2. **Workload infra label** comes from the cluster *name* (`"slurm"` in the name -> SLURM, else K8s), so `helios` shows as K8s
   in workload health. Suggested change: take it from the cluster entity type or the snapshot's `infra_source`.
3. **Agent status is time-based** (5 minutes). A mock org needs `--heartbeat` while being tested, or a scheduled re-run.

## Check without the UI

The backend repo has `scripts/validate_mocks` that calls the same service functions directly:
set `MONGO_URI`, `ORG_ID=6679f5a9-398f-4b58-b04d-e5d263ec2039`, `ORG_SLUG=pxz7ru3sqvx`, then `go run ./scripts/validate_mocks`
in `C:\Users\Ctr_Sriraam\wrench-service\wrench-service`. It prints clusters / standalone counts, resources with agentStatus,
agent health and trend points - the same values as the table above.
