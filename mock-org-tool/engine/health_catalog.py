"""Metric catalogs for workload-health.json. Inference mirrors the reference; TRAINING_JOB is invented."""


def U(lo, hi):
    return lambda r, ctx: round(r.uniform(lo, hi), 1)


def C(v):
    return lambda r, ctx: v


def M(metric_id, name, unit, critical, weight, value, score=100):
    return {"metricId": metric_id, "metricName": name, "unit": unit, "isCritical": critical,
            "weight": weight, "value": value, "score": score}


def CAT(category_id, name, description, weight, metrics):
    return {"categoryId": category_id, "categoryName": name, "categoryDescription": description,
            "effectiveWeight": weight, "metrics": metrics}


# Healthy values below come from the member GPUs of the workload (ctx) or the reference's healthy range.
_GPU_HARDWARE_METRICS = [
    M("gpu_temperature", "GPU Temperature (\u00b0C)", "celsius", True, 0, lambda r, ctx: ctx["max_temp"]),
    M("ecc_uncorrectable", "ECC Uncorrectable Error Count (DBE)", "count", True, 0, C(0)),
    M("ecc_correctable_rate", "ECC Correctable Error Rate (SBE/s)", "count_per_second", False, 1, C(0)),
    M("xid_errors", "Xid Error Count", "count", True, 0, C(0)),
    M("row_remap_failure", "Row-Remap Failure Count", "count", True, 0, C(0)),
]

INFERENCE = [
    CAT("user_experience_latency", "User Experience (Latency)",
        "End-user latency signals: TTFT, TPOT and queue depth", 3, [
            M("ttft", "TTFT \u2014 Time to First Token (ms, p95)", "ms", False, 3, U(145.7, 172.3)),
            M("tpot", "TPOT \u2014 Time Per Output Token (ms/token, p95)", "ms", False, 3, U(17.7, 20.4)),
            M("queue_wait_proxy", "Request Queue Depth (waiting count proxy)", "count", False, 2, U(5.1, 6.0)),
        ]),
    CAT("resource_cache_efficiency", "Resource & Cache Efficiency",
        "KV-cache utilisation, request concurrency and GPU VRAM pressure", 3, [
            M("gpu_kv_cache_pct", "GPU KV Cache Usage %", "percent", False, 3, U(57.8, 69.7)),
            M("cpu_kv_cache_pct", "CPU KV Cache Usage % (Swapped)", "percent", False, 1, U(0.9, 1.1)),
            M("requests_waiting", "Requests Waiting (Queue Depth)", "count", False, 2, U(3.2, 3.8)),
            M("gpu_memory_vram", "GPU Memory (VRAM) Used %", "percent", True, 0, lambda r, ctx: ctx["mem_pct"]),
        ]),
    CAT("model_quality", "Model Quality",
        "Inference failure rate and non-stop finish reasons indicating model output quality issues", 3, [
            M("inference_failure_count", "Inference Request Failure Count", "count_per_second", True, 0, C(0)),
            M("non_stop_finish_rate", "Non-\"stop\" Finish Reason Rate %", "percent", False, 2, U(2.1, 2.5)),
        ]),
    CAT("throughput_batching_efficiency", "Throughput & Batching Efficiency",
        "Compute vs queue time ratio indicating how efficiently GPU capacity serves requests", 2, [
            M("compute_vs_queue_ratio", "Compute vs Queue Time Ratio", "ratio", False, 2, U(1.0, 1.2)),
        ]),
    CAT("gpu_hardware_health", "GPU Hardware Health",
        "Thermal, electrical and memory integrity of inference GPUs under continuous serving load", 3,
        _GPU_HARDWARE_METRICS),
    CAT("fault_tolerance_reliability", "Fault Tolerance & Reliability",
        "Pod restarts, container terminations and node readiness for inference serving continuity", 3, [
            M("pod_restart_count", "Pod / Container Restart Count", "count", True, 0, C(0)),
            M("container_termination", "Container Termination (OOMKilled / Error)", "count", True, 0, C(0)),
            M("node_readiness", "Node Readiness", "count", True, 0, C(0)),
        ]),
    CAT("network_load_balancing", "Network & Load Balancing",
        "Host network errors and container-level throughput on serving nodes", 1, [
            M("host_network_errors", "Host Network Errors", "count_per_second", False, 2, C(0)),
            M("container_network_throughput", "Container Network Throughput (bytes/s)", "bytes_per_second",
              False, 0, U(74714126.2, 91122911.7), score=-1),
        ]),
    CAT("advanced_serving_optimizations", "Advanced Serving Optimizations",
        "KV-cache eviction and request preemption rate; nonzero means GPU is over-provisioned relative to VRAM",
        2, [
            M("kv_cache_eviction_rate", "KV Cache Eviction / Preemption Rate", "count_per_second", False, 2,
              C(0.3)),
        ]),
]

TRAINING = [
    CAT("training_throughput", "Training Throughput",
        "Token throughput, step time and model FLOPs utilisation of the training job", 3, [
            M("tokens_per_second_per_gpu", "Tokens per Second per GPU", "tokens_per_second", False, 3,
              U(2800.0, 3600.0)),
            M("step_time", "Step Time (s)", "seconds", False, 2, U(2.8, 3.6)),
            M("mfu_pct", "Model FLOPs Utilization (MFU) %", "percent", False, 3, U(38.0, 46.0)),
        ]),
    CAT("training_convergence", "Training Convergence",
        "Loss stability: NaN/Inf losses, loss spikes and gradient norm", 3, [
            M("loss_nan_count", "NaN / Inf Loss Count", "count", True, 0, C(0)),
            M("loss_spike_count", "Loss Spike Count", "count", False, 2, C(0)),
            M("gradient_norm", "Gradient Norm", "norm", False, 1, U(0.4, 1.1)),
        ]),
    CAT("checkpoint_reliability", "Checkpointing & Recovery",
        "Checkpoint freshness, write duration and failures", 2, [
            M("checkpoint_age", "Time Since Last Checkpoint (min)", "minutes", False, 2, U(8.0, 24.0)),
            M("checkpoint_write_duration", "Checkpoint Write Duration (s)", "seconds", False, 1, U(35.0, 80.0)),
            M("checkpoint_failure_count", "Checkpoint Failure Count", "count", True, 0, C(0)),
        ]),
    CAT("distributed_communication", "Distributed Communication (NCCL)",
        "Collective-communication overhead and errors across ranks", 3, [
            M("comm_time_pct", "Communication Time % of Step", "percent", False, 3, U(8.0, 15.0)),
            M("nccl_error_count", "NCCL Error Count", "count", True, 0, C(0)),
            M("network_retransmits", "Network Retransmits", "count_per_second", False, 1, C(0)),
        ]),
    CAT("data_pipeline", "Data Pipeline",
        "Dataloader stalls and dataset read throughput", 2, [
            M("dataloader_wait_pct", "Dataloader Wait % of Step", "percent", False, 2, U(1.0, 4.0)),
            M("dataset_read_throughput", "Dataset Read Throughput (bytes/s)", "bytes_per_second", False, 0,
              U(1.2e9, 2.4e9), score=-1),
        ]),
    CAT("gpu_hardware_health", "GPU Hardware Health",
        "Thermal, electrical and memory integrity of training GPUs under sustained load", 3,
        _GPU_HARDWARE_METRICS),
    CAT("job_reliability", "Job Reliability (Slurm)",
        "Requeues, node failures and OOM kills affecting the running job", 3, [
            M("job_requeue_count", "Job Requeue Count", "count", True, 0, C(0)),
            M("node_failure_count", "Node Failure Count", "count", True, 0, C(0)),
            M("oom_killed_count", "OOM Killed Count", "count", True, 0, C(0)),
        ]),
]

CATALOG = {"INFERENCE_SERVICE": INFERENCE, "TRAINING_JOB": TRAINING}
