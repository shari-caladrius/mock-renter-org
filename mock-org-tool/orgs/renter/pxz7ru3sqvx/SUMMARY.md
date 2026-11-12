# yedol22330.jcaton.com (RENTER, slug pxz7ru3sqvx, snapshot 2026-11-12T16:35:05Z)

GPU clusters **3** (1 SLURM, 2 K8s, 160 GPUs, 20 instances)  
GPU fleet utilisation **36.4%**, 128 of 160 GPUs utilised; avg 54.1 °C, 316 W, memory 53.7%  
Storage **46%** (204.2 / 444 TiB)  
Fleet cost **$638.40/hr**, $15,321.60/day  
GPU efficiency: idle 32 (20%), under 48 (30%), normal 64 (40%), over 16 (10%)  
Workload health (22): 22 healthy, 0 warning, 0 critical

| Cluster | Infra | GPU | Instances | GPUs | Training | Inference | Avg util % | Temp °C (busy) | Power W (busy) |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| helios | Slurm | H200 | 8 | 64 | 3 | 4 | 42.9 | 40-83 | 98-700 |
| atlas | K8s | H200 | 6 | 48 | 1 | 6 | 27 | 40-83 | 98-700 |
| orion | K8s | H200 | 6 | 48 | 2 | 6 | 37.3 | 40-83 | 98-700 |
