#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, statistics, subprocess, shlex, re, sys, os, csv, datetime
from typing import Dict, List, Tuple, Optional

try:
    import psutil
except ImportError:
    print("請先安裝 psutil：  python3 -m pip install psutil", file=sys.stderr)
    sys.exit(1)

def run(cmd: str, timeout: int = 5, force_c_locale: bool = False) -> str:
    try:
        env = os.environ.copy()
        if force_c_locale:
            env["LC_ALL"] = "C"
            env["LANG"] = "C"
        out = subprocess.check_output(shlex.split(cmd), timeout=timeout,
                                      stderr=subprocess.DEVNULL, env=env)
        return out.decode("utf-8", "ignore").strip()
    except Exception:
        return ""

def mean_or_na(vals: List[float], ndigits:int=1) -> Optional[float]:
    return round(statistics.mean(vals), ndigits) if vals else None

def minmax_or_na(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    return (min(vals), max(vals)) if vals else (None, None)

def get_battery_state(lang: str = "zh") -> str:
    devs = run("upower -e", force_c_locale=True)
    bat = ""
    for line in devs.splitlines():
        if "battery" in line.lower():
            bat = line.strip(); break
    if not bat:
        return "N/A"
    info = run(f"upower -i {bat}", force_c_locale=True)
    m = re.search(r"state:\s*(\S+)", info)
    state = (m.group(1) if m else "").lower()
    if lang == "en":
        mapping = {"discharging":"discharging","charging":"charging","fully-charged":"fully-charged"}
    else:
        mapping = {"discharging":"放電","charging":"充電","fully-charged":"已充滿"}
    return mapping.get(state, "未知" if lang != "en" else "unknown")

def get_cpu_usage_sample() -> float:
    return psutil.cpu_percent(interval=None)

def nvidia_smi_available() -> bool:
    return bool(run("which nvidia-smi"))

def get_gpu_stats(gpu_index:int=0) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    透過 nvidia-smi 取得 (GPU使用率%, GPU溫度°C, GPU風扇 文字)
    風扇多數筆電會回 'N/A'；桌機通常回百分比（0–100）
    """
    if not nvidia_smi_available():
        return (None, None, None)
    cmd = f"nvidia-smi -i {gpu_index} --query-gpu=utilization.gpu,temperature.gpu,fan.speed --format=csv,noheader,nounits"
    q = run(cmd, force_c_locale=True)
    if not q:
        return (None, None, None)
    line = q.splitlines()[0].strip()
    parts = [p.strip() for p in line.split(",")]
    util = None
    temp = None
    fan_text = None
    try:
        if len(parts) > 0 and parts[0] and parts[0].upper() != "N/A":
            util = float(parts[0])
        if len(parts) > 1 and parts[1] and parts[1].upper() != "N/A":
            temp = float(parts[1])
        if len(parts) > 2:
            fan_text = parts[2] if parts[2] else "N/A"
            fan_text = "N/A" if fan_text.upper() == "N/A" else f"{fan_text} %"
    except Exception:
        pass
    return (util, temp, fan_text)

def parse_sensors() -> Dict[str, float]:
    """
    從 `sensors` 嘗試抓 CPU core 溫度與 CPU 風扇（若平台支援）。
    抓不到沒關係，回傳空 dict，最後輸出 N/A。
    """
    txt = run("sensors")
    res: Dict[str, float] = {}
    if not txt:
        return res
    for line in txt.splitlines():
        line = line.strip()
        m = re.search(r'(?i)(cpu).*fan.*?:\s*([0-9]+)\s*RPM', line)
        if m:
            res["cpu_fan_rpm"] = float(m.group(2)); continue
        m = re.search(r'(?i)\bfan[0-9]*:\s*([0-9]+)\s*RPM', line)
        if m and "cpu_fan_rpm" not in res:
            res["cpu_fan_rpm"] = float(m.group(1)); continue
        m = re.search(r'(?i)\bcore\s*[0-9]+:\s*\+?([0-9]+(?:\.[0-9]+)?)°?C', line)
        if m:
            res.setdefault("cpu_core_temps", [])
            res["cpu_core_temps"].append(float(m.group(1))); continue
        m = re.search(r'(?i)\b(Tctl|Tdie):\s*\+?([0-9]+(?:\.[0-9]+)?)°?C', line)
        if m:
            res.setdefault("cpu_core_temps", [])
            res["cpu_core_temps"].append(float(m.group(2))); continue
    return res

def get_cpu_freq_mhz() -> Optional[float]:
    freqs = psutil.cpu_freq(percpu=True)
    if not freqs:
        return None
    vals = [f.current for f in freqs if f]
    return statistics.mean(vals) if vals else None

def match_proc_cpu_percent(pattern: str) -> float:
    pat = pattern.lower()
    total = 0.0
    for p in psutil.process_iter(attrs=["name", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmdl = " ".join(p.info.get("cmdline") or []).lower()
            if pat in name or pat in cmdl:
                total += p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total

def warmup_cpu_percent():
    psutil.cpu_percent(interval=None)
    for p in psutil.process_iter():
        try:
            p.cpu_percent(interval=None)
        except Exception:
            pass

def write_csv_header(path: str, fieldnames: List[str]):
    need_header = True
    if os.path.exists(path) and os.path.getsize(path) > 0:
        need_header = False
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            w.writeheader()

def append_csv_rows(path: str, fieldnames: List[str], rows: List[Dict[str, object]]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser(description="收集系統效能指標；逐偵紀錄並存 CSV（含最終平均；GPU 使用 nvidia-smi）")
    ap.add_argument("--label", required=True, help="場景標籤（例如：空載/未貼皮/貼皮）")
    ap.add_argument("--duration", type=float, default=20.0, help="量測秒數（預設 20s）")
    ap.add_argument("--interval", type=float, default=1.0, help="取樣間隔秒（預設 1s）")
    ap.add_argument("--gpu-index", type=int, default=0, help="nvidia-smi GPU index（預設 0）")
    ap.add_argument("--proc", action="append", default=[],
                   help='追蹤特定行程 CPU%%，格式 label=pattern；例如： --proc "python3=python3" --proc "MegaPoseClient=megaposeclient"')
    ap.add_argument("--csv", type=str, default=None, help="CSV 輸出檔名（未指定則自動命名）")
    ap.add_argument("--csv-append", action="store_true", help="若存在則追加（不覆蓋）")
    ap.add_argument("--battery-en", action="store_true", help="以英文寫入電池狀態（避免亂碼）")
    args = ap.parse_args()

    # 解析 --proc
    proc_specs: List[Tuple[str,str]] = []
    for item in args.proc:
        if "=" in item:
            k, v = item.split("=", 1)
            proc_specs.append((k.strip(), v.strip()))
        else:
            proc_specs.append((item.strip(), item.strip()))

    # CSV 檔名
    if args.csv:
        csv_path = args.csv
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = re.sub(r"[^\w\-]+", "_", args.label)
        csv_path = f"perf_{safe_label}_{ts}.csv"

    warmup_cpu_percent()

    # 累積樣本（用於平均）
    cpu_usage_samples: List[float] = []
    cpu_freq_samples: List[float] = []
    cpu_core_temps_all: List[float] = []
    frame_cpu_temp_min: List[float] = []
    frame_cpu_temp_max: List[float] = []
    cpu_fan_samples: List[float] = []
    gpu_usage_samples: List[float] = []
    gpu_temp_samples: List[float] = []
    gpu_fan_texts: List[str] = []
    proc_cpu_series: Dict[str, List[float]] = {lbl: [] for (lbl, _) in proc_specs}

    # CSV 逐偵資料
    csv_rows: List[Dict[str, object]] = []

    t_start = time.time()
    battery_state = get_battery_state("en" if args.battery_en else "zh")

    # 先嘗試 sensors（可沒有）
    sx0 = parse_sensors()
    if isinstance(sx0.get("cpu_core_temps"), list):
        cpu_core_temps_all.extend(sx0["cpu_core_temps"])
    if "cpu_fan_rpm" in sx0:
        cpu_fan_samples.append(float(sx0["cpu_fan_rpm"]))

    while True:
        now = time.time()
        elapsed = now - t_start
        if elapsed >= args.duration:
            break

        # CPU
        cpu_usage = get_cpu_usage_sample()
        cpu_usage_samples.append(cpu_usage)

        cpu_freq = get_cpu_freq_mhz()
        if cpu_freq is not None:
            cpu_freq_samples.append(cpu_freq)

        sx = parse_sensors()
        temps_sample = sx.get("cpu_core_temps", [])
        if isinstance(temps_sample, list) and temps_sample:
            cpu_core_temps_all.extend(temps_sample)
            frame_cpu_temp_min.append(min(temps_sample))
            frame_cpu_temp_max.append(max(temps_sample))
            cpu_temp_min = min(temps_sample)
            cpu_temp_max = max(temps_sample)
        else:
            cpu_temp_min = None
            cpu_temp_max = None

        cpu_fan = sx.get("cpu_fan_rpm", None)
        if cpu_fan is not None:
            f = float(cpu_fan)
            cpu_fan_samples.append(f)
            cpu_fan = int(f)

        # GPU
        g_util, g_temp, g_fan_text = get_gpu_stats(args.gpu_index)
        if g_util is not None: gpu_usage_samples.append(g_util)
        if g_temp is not None: gpu_temp_samples.append(g_temp)
        if g_fan_text is not None: gpu_fan_texts.append(g_fan_text)

        # Processes（逐偵）
        proc_sample_vals: Dict[str, float] = {}
        for lbl, pat in proc_specs:
            v = match_proc_cpu_percent(pat)
            proc_sample_vals[lbl] = v
            proc_cpu_series[lbl].append(v)

        # 逐偵 CSV 列
        csv_row: Dict[str, object] = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "label": args.label,
            "elapsed_s": round(elapsed, 3),
            "battery": battery_state,
            "cpu_usage_percent": round(cpu_usage, 2) if cpu_usage is not None else "N/A",
            "cpu_freq_mhz": round(cpu_freq, 4) if cpu_freq is not None else "N/A",
            "cpu_fan_rpm": cpu_fan if cpu_fan is not None else "N/A",
            "cpu_temp_min_c": round(cpu_temp_min, 1) if cpu_temp_min is not None else "N/A",
            "cpu_temp_max_c": round(cpu_temp_max, 1) if cpu_temp_max is not None else "N/A",
            "gpu_util_percent": round(g_util, 2) if g_util is not None else "N/A",
            "gpu_temp_c": round(g_temp, 1) if g_temp is not None else "N/A",
            "gpu_fan": g_fan_text if g_fan_text is not None else "N/A",
        }
        for lbl in [k for k,_ in proc_specs]:
            v = proc_sample_vals.get(lbl, None)
            csv_row[f"{lbl}_cpu_percent"] = round(v, 2) if v is not None else "N/A"

        csv_rows.append(csv_row)
        time.sleep(max(0.0, args.interval))

    # ====== 彙整平均 ======
    cpu_temp_global_min, cpu_temp_global_max = minmax_or_na(cpu_core_temps_all)
    cpu_fan_avg = mean_or_na(cpu_fan_samples, 0)
    cpu_usage_avg = mean_or_na(cpu_usage_samples, 2)
    cpu_freq_avg  = mean_or_na(cpu_freq_samples, 4)
    gpu_util_avg  = mean_or_na(gpu_usage_samples, 2)
    gpu_temp_avg  = mean_or_na(gpu_temp_samples, 1)

    # GPU 風扇摘要（最後一次非 N/A）
    gpu_fan_cell = "N/A"
    for x in reversed(gpu_fan_texts):
        if x and x.upper() != "N/A":
            gpu_fan_cell = x
            break

    proc_means = {}
    for lbl, _ in proc_specs:
        series = proc_cpu_series.get(lbl, [])
        proc_means[lbl] = mean_or_na(series, 2)

    # --- Markdown 摘要（仍會列印） ---
    cpu_temp_range_txt = "N/A" if cpu_temp_global_min is None else f"{cpu_temp_global_min:.0f}–{cpu_temp_global_max:.0f} °C"
    header = ["項目","電池","CPU溫度(每核心)","GPU溫度","CPU風扇速度","GPU風扇速度",
              "CPU使用率","GPU使用率"] + [f"{lbl} CPU使用率" for (lbl,_) in proc_specs] + ["CPU頻率"]
    row = [
        args.label,
        battery_state,
        cpu_temp_range_txt,
        f"{gpu_temp_avg:.0f} °C" if gpu_temp_avg is not None else "N/A",
        f"{cpu_fan_avg:.0f} RPM" if cpu_fan_avg is not None else "N/A",
        gpu_fan_cell,
        f"{cpu_usage_avg:.1f} % " if cpu_usage_avg is not None else "N/A",
        f"{gpu_util_avg:.1f} % " if gpu_util_avg is not None else "N/A",
        *[(f"{proc_means[lbl]:.2f} %" if proc_means[lbl] is not None else "N/A") for (lbl,_) in proc_specs],
        f"{cpu_freq_avg:.4f} MHz" if cpu_freq_avg is not None else "N/A",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"]*len(header)) + "|")
    print("| " + " | ".join(row) + " |")

    # --- CSV 輸出（逐偵 + 平均） ---
    base_fields = ["timestamp","label","elapsed_s","battery",
                   "cpu_usage_percent","cpu_freq_mhz","cpu_fan_rpm",
                   "cpu_temp_min_c","cpu_temp_max_c",
                   "gpu_util_percent","gpu_temp_c","gpu_fan"]
    proc_fields = [f"{lbl}_cpu_percent" for (lbl,_) in proc_specs]
    fieldnames = base_fields + proc_fields

    # 若沒有 --csv-append 且檔案已存在，覆蓋
    if (not args.csv_append) and os.path.exists(csv_path):
        os.remove(csv_path)

    # 寫表頭 + 逐偵資料
    write_csv_header(csv_path, fieldnames)
    append_csv_rows(csv_path, fieldnames, csv_rows)

    # 追加「平均」列（與逐偵同欄位）
    avg_row: Dict[str, object] = {
        "timestamp": "SUMMARY",
        "label": f"{args.label}_avg",
        "elapsed_s": round(time.time() - t_start, 3),
        "battery": battery_state,
        "cpu_usage_percent": cpu_usage_avg if cpu_usage_avg is not None else "N/A",
        "cpu_freq_mhz": cpu_freq_avg if cpu_freq_avg is not None else "N/A",
        # 平均列的 CPU 風扇用平均；溫度改放整體 min/max（更符合「區間」語義）
        "cpu_fan_rpm": int(cpu_fan_avg) if cpu_fan_avg is not None else "N/A",
        "cpu_temp_min_c": round(cpu_temp_global_min, 1) if cpu_temp_global_min is not None else "N/A",
        "cpu_temp_max_c": round(cpu_temp_global_max, 1) if cpu_temp_global_max is not None else "N/A",
        "gpu_util_percent": gpu_util_avg if gpu_util_avg is not None else "N/A",
        "gpu_temp_c": gpu_temp_avg if gpu_temp_avg is not None else "N/A",
        "gpu_fan": gpu_fan_cell,
    }
    for lbl, _ in proc_specs:
        avg_row[f"{lbl}_cpu_percent"] = proc_means[lbl] if proc_means[lbl] is not None else "N/A"

    append_csv_rows(csv_path, fieldnames, [avg_row])
    print(f"\nCSV 已輸出（含平均列）：{csv_path}")

if __name__ == "__main__":
    main()

# 安裝依賴
# python3 -m pip install --upgrade psutil

# 例：foundationpose 場景，逐偵 + 平均一起寫入 CSV（英文電池狀態避免亂碼）
# ./collect_perf.py --label foundationpose --gpu-index 0 --duration 30 --interval 1 --proc "python3=foundationpose_" --battery-en --csv perf_foundationpose.csv

# 例：megapose 追加到同一檔
# ./collect_perf.py --label megapose貼皮 --gpu-index 0 --duration 60 --interval 1 --proc "python3=foundationpose_" --proc "MegaPoseClient=megaposeclient" --battery-en --csv perf_foundationpose.csv --csv-append

