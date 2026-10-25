"""Compare the same browser renderer/GPU processes against their unloaded baseline."""
import json
from pathlib import Path
import sys

folder = Path(sys.argv[1])
pids = {int(value) for value in sys.argv[2:]}
assert pids, 'Pass the measured page renderer and GPU process IDs.'
frames = [json.loads(line) for line in (folder / 'memory-samples.jsonl').read_text().splitlines()]
assert len(frames) > 100, 'Record setup and generation before summarizing.'


def totals(frame):
    processes = [p for p in frame['processes'] if p['pid'] in pids]
    assert {p['pid'] for p in processes} == pids, 'A measured browser process exited or changed ID.'
    gpu = [entry for p in processes for entry in p['gpu']]
    return {'privateCommitBytes': sum(p['privateCommitBytes'] for p in processes),
            'gpuDedicatedCounterBytes': sum(g['dedicatedBytes'] for g in gpu),
            'gpuSharedCounterBytes': sum(g['sharedBytes'] for g in gpu),
            'gpuCombinedCounterBytes': sum(g['dedicatedBytes'] + g['sharedBytes'] for g in gpu)}


baseline = totals(frames[0])
peaks = {key: max(totals(f)[key] for f in frames) for key in baseline}
unloaded = json.loads((folder / 'baseline-uss.json').read_text())
ready = json.loads((folder / 'ready-uss.json').read_text())
assert {p['pid'] for p in unloaded} == {p['pid'] for p in ready} == pids
physical = sum(p['uss'] for p in ready) - sum(p['uss'] for p in unloaded)
assert physical > 0
report = {'method': 'Windows private physical RAM (USS) snapshots and ~4 Hz private commit / PDH GPU process counters, baseline subtracted.',
          'limitations': ['Measured on a 64 GB RAM laptop; this is not an 8 GB hardware certification. Other GPU activity was not controlled.',
                         'GPU counters and system RAM overlap; do not add them as independent physical memory.',
                         'USS is measured at ready state; sampling can miss short peaks. Private commit includes pagefile backing.'],
          'pids': sorted(pids), 'sampleCount': len(frames), 'durationSeconds': frames[-1]['unixTime'] - frames[0]['unixTime'],
          'baseline': baseline, 'sampledPeak': peaks,
          'sampledPeakIncrease': {key: peaks[key] - baseline[key] for key in baseline},
          'readyPrivatePhysicalIncreaseBytes': physical}
physical_frames = []
for frame in frames:
    processes = [p for p in frame['processes'] if p['pid'] in pids]
    if all('privatePhysicalBytes' in p for p in processes):
        ram = max(0, sum(p['privatePhysicalBytes'] for p in processes) - sum(p['uss'] for p in unloaded))
        gpu = max(0, totals(frame)['gpuCombinedCounterBytes'] - baseline['gpuCombinedCounterBytes'])
        physical_frames.append((ram, ram + gpu))
if physical_frames:
    report.update(method='Timestamped Windows process / GPU counters with private physical RAM (USS) sampled about once per second.',
                  privatePhysicalSampleCount=len(physical_frames),
                  sampledPrivatePhysicalIncreaseBytes=max(p[0] for p in physical_frames),
                  sampledPhysicalPlusGpuIncreaseBytes=max(p[1] for p in physical_frames))
    report['limitations'][2] = 'Sampled peaks can miss short spikes. RAM plus GPU is a conservative planning sum with overlap, not a measured physical total. Commit includes pagefile backing.'
(folder / 'summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
