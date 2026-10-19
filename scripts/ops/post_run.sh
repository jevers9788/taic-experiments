#!/usr/bin/env bash
# On the GPU VM, after a TAIC run: pick TAIC's accepted answer (else the fastest
# verified candidate), verify it independently, racecheck it, and pack results.
# Usage: GPU=v100 KERNEL=... TAG=... [PTX_LAUNCHER_KERNELS=...] bash scripts/ops/post_run.sh
# Run from the experiment dir on the VM.
cd "$(dirname "$0")/../.." && source scripts/env.sh
GPU=${GPU:-v100}
D=$(ls -td runs/*_${KERNEL}_* | head -1)
SEL=$(.venv/bin/python - "$D" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
final = glob.glob(os.path.join(d, 'final_speedup_vs_triton_*.json'))
if final:
    print('accepted', os.path.join(d, 'final_candidate.ptx'), json.load(open(final[0]))['num_threads_x'])
    sys.exit()
best = None
for j in glob.glob(os.path.join(d, 'tool_output', '*_launch_verifier.json')):
    r = json.load(open(j)); a = r.get('answer') or {}
    ptx = glob.glob(j.replace('.json', '_speedup_vs_triton_*.ptx'))
    if a.get('passed') and ptx and (best is None or a['speedup_vs_triton'] > best[0]):
        best = (a['speedup_vs_triton'], ptx[0], r['resulting_payload']['num_threads_x'])
print('best_tested', best[1], best[2]) if best else print('none')
PY
)
echo "run: $D" > logs/$GPU-$TAG-selection.log; echo "selection: $SEL" >> logs/$GPU-$TAG-selection.log
read KIND PTX THREADS <<< "$SEL"
T=results/$GPU/$KERNEL.json
step() { local n=$1; shift; { echo "Command: $*"; echo "Start: $(date -u +%FT%TZ)"; } > logs/$GPU-$n.log; "$@" >> logs/$GPU-$n.log 2>&1; echo "Exit: $? $(date -u +%FT%TZ)" >> logs/$GPU-$n.log; }
if [[ $KIND != none ]]; then
  step $TAG-verify .venv/bin/python scripts/verify_candidate.py $KERNEL "$PTX" --threads-x "$THREADS" --tuning-from $T --output results/$GPU/$TAG-verify.json
  step $TAG-racecheck .venv/bin/python scripts/racecheck.py $KERNEL "$PTX" --threads-x "$THREADS" --tuning-from $T --timeout 3600
fi
O=/tmp/out-$TAG; rm -rf $O && mkdir -p $O/runs $O/logs $O/results
cp -r "$D" $O/runs/; cp logs/$GPU-$TAG*.log $O/logs/ 2>/dev/null; cp results/$GPU/$TAG-verify.json $O/results/ 2>/dev/null
chmod -R a+rX $O; tar czf $O.tgz -C /tmp out-$TAG; echo DONE > $O.done
# Backstop: stop the VM ourselves even if the local driver is asleep (copy-back can
# still happen after a restart; results stay on the disk).
sudo shutdown -h +20 'post_run done; stopping in 20 min' 2>/dev/null || true
