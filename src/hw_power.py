"""Power logger for the hardware sweeps: samples the CPU package power (Windows Energy Meter / RAPL: cores + integrated GPU + NPU share the package) once a
second and the discrete GPU's board power every 200 ms, timestamped, into results/hardware/power_*.csv until it is stopped (Ctrl-C or the `stop` file).

    uv run python src/hw_power.py --out results/hardware &     # start, run the sweeps, then `touch results/hardware/stop`

src/hw_report.py averages the samples that fall inside each configuration's start/end window and subtracts the idle package power (a quiet period at
the start of the log) to get energy per sequence. Package power is the whole package, so the number includes memory-controller and fabric activity the
workload causes; it does not include DRAM chips or the board."""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/hardware")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stop = out / "stop"
    stop.unlink(missing_ok=True)
    cpu = subprocess.Popen(["typeperf", r"\Energy Meter(RAPL_Package0_PKG)\Power", "-si", "1"], stdout=open(out / "power_package.csv", "w"), stderr=subprocess.DEVNULL)  # noqa: SIM115
    gpu = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,power.draw", "--format=csv,noheader,nounits", "-lms", "200"], stdout=open(out / "power_gpu.csv", "w"),  # noqa: SIM115
                           stderr=subprocess.DEVNULL)
    print(f"[power] logging to {out} (touch {stop} to stop)", flush=True)
    try:
        while not stop.exists():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cpu.terminate()
        gpu.terminate()
    print("[power] stopped", flush=True)


if __name__ == "__main__":
    sys.exit(main())
