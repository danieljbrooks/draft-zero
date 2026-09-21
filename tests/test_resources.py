"""Container limits. A pod reports the HOST's cpu/ram; only cgroup tells the truth."""
from draftzero import resources


def test_quota_helpers_are_safe_without_cgroup():
    # On macOS these must return None rather than raising: the same code runs locally.
    for fn in (resources.cpu_quota, resources.mem_limit_gb,
               resources.cgroup_mem_used_gb, resources.cgroup_mem_failcnt):
        fn()


def test_monitor_collects_samples():
    m = resources.ResourceMonitor(interval=0.05).start()
    import time
    time.sleep(0.3)
    m.stop()
    s = m.summary()
    assert s["samples"] > 0
    assert "cpu_percent_mean" in s
    assert "ram_total_gb" in s


def test_summary_is_empty_without_samples():
    assert resources.ResourceMonitor().summary()["samples"] == 0
