"""Read Windows browser RAM and GPU counters during a manually labelled run."""
import ctypes as ct
from ctypes import wintypes as wt
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil

OUT = Path(__file__).resolve().parents[1] / 'artifacts'
if len(sys.argv) > 1:
    OUT /= sys.argv[1]
OUT.mkdir(exist_ok=True)
PHASE = OUT / 'memory-phase.txt'
PDH = ct.WinDLL('pdh')


class CounterNumber(ct.Union):
    _fields_ = [('large', ct.c_longlong), ('double', ct.c_double)]


class CounterValue(ct.Structure):
    _fields_ = [('status', wt.DWORD), ('number', CounterNumber)]


class CounterItem(ct.Structure):
    _fields_ = [('name', wt.LPWSTR), ('value', CounterValue)]


PDH.PdhOpenQueryW.argtypes = [wt.LPCWSTR, ct.c_size_t, ct.POINTER(wt.HANDLE)]
PDH.PdhAddEnglishCounterW.argtypes = [wt.HANDLE, wt.LPCWSTR, ct.c_size_t, ct.POINTER(wt.HANDLE)]
PDH.PdhCollectQueryData.argtypes = [wt.HANDLE]
PDH.PdhGetFormattedCounterArrayW.argtypes = [wt.HANDLE, wt.DWORD, ct.POINTER(wt.DWORD), ct.POINTER(wt.DWORD), ct.c_void_p]
PDH.PdhCloseQuery.argtypes = [wt.HANDLE]


def check(code):
    if code:
        raise OSError(f'PDH status 0x{code & 0xffffffff:08x}')


def values(counter):
    size, count = wt.DWORD(), wt.DWORD()
    code = PDH.PdhGetFormattedCounterArrayW(counter, 0x400, ct.byref(size), ct.byref(count), None)
    if (code & 0xffffffff) != 0x800007D2:
        check(code)
    buffer = ct.create_string_buffer(size.value)
    check(PDH.PdhGetFormattedCounterArrayW(counter, 0x400, ct.byref(size), ct.byref(count), buffer))
    items = ct.cast(buffer, ct.POINTER(CounterItem))
    result = {}
    for i in range(count.value):
        item = items[i]
        if item.value.status not in (0, 1):
            continue
        match = re.match(r'pid_(\d+)_(.*)', item.name)
        if match:
            key = (int(match[1]), match[2])
            result[key] = item.value.number.large
    assert result and all(amount >= 0 for amount in result.values()), 'GPU counters must contain valid nonnegative readings.'
    return result


def sample(query, counters, physical_pids):
    check(PDH.PdhCollectQueryData(query))
    dedicated, shared = [values(counter) for counter in counters]
    processes = []
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'].lower() not in ('chatgpt.exe', 'codex.exe'):
            continue
        try:
            memory = process.memory_info()
            assert memory.private > 0 and memory.rss > 0
            gpu = [{'adapter': adapter, 'dedicatedBytes': value, 'sharedBytes': shared.get((pid, adapter), 0)}
                   for (pid, adapter), value in dedicated.items() if pid == process.pid]
            row = {'pid': process.pid, 'name': process.info['name'], 'residentBytes': memory.rss,
                   'privateCommitBytes': memory.private, 'gpu': gpu}
            if process.pid in physical_pids:
                row['privatePhysicalBytes'] = process.memory_full_info().uss
            processes.append(row)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    assert processes, 'No browser processes were sampled.'
    return {'unixTime': time.time(), 'phase': PHASE.read_text().strip(),
            'systemAvailableBytes': psutil.virtual_memory().available, 'processes': processes}


def main():
    query = wt.HANDLE()
    check(PDH.PdhOpenQueryW(None, 0, ct.byref(query)))
    counters = []
    for kind in ('Dedicated Usage', 'Shared Usage'):
        counter = wt.HANDLE()
        check(PDH.PdhAddEnglishCounterW(query, '\\GPU Process Memory(*)\\' + kind, 0, ct.byref(counter)))
        counters.append(counter)
    frames = []
    baseline_file = OUT / 'baseline-uss.json'
    physical_pids = {p['pid'] for p in json.loads(baseline_file.read_text())} if baseline_file.exists() else set()
    next_physical = 0
    with (OUT / 'memory-nvidia.csv').open('w') as gpu_file:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,memory.free',
                                    '--format=csv,noheader,nounits', '-lms', '250'], stdout=gpu_file,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline and PHASE.read_text().strip() != 'stop':
                now = time.monotonic()
                measure_physical = now >= next_physical
                frame = sample(query, counters, physical_pids if measure_physical else set())
                if measure_physical:
                    next_physical = time.monotonic() + 1
                frames.append(frame)
                with (OUT / 'memory-samples.jsonl').open('a') as output:
                    output.write(json.dumps(frame) + '\n')
                if len(frames) == 1:
                    print(f"Memory sampler ready: {len(frame['processes'])} browser/app processes.", flush=True)
                time.sleep(.25)
        finally:
            monitor.terminate()
            monitor.wait()
            PDH.PdhCloseQuery(query)
    print(f'Memory sampling complete: {len(frames)} samples.', flush=True)


if __name__ == '__main__':
    main()
