#!/usr/bin/env bash
# sweep15.sh — drafting keepers on varied chat prompts (tools/chatbench.py) through the live server: default vs --mtp-q4 proj vs --spec-min-p 0.35, 2 rounds, then the default server back on
cd ~/work/rdna3
up() { ./strata-rdna3.sh stop >/dev/null 2>&1; XARGS="$1" ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1;
       local j; for j in $(seq 600); do curl -sf 127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2; done; }
{
for i in 1 2; do for x in "base|" "q4p|--mtp-q4 proj" "p35|--spec-min-p 0.35"; do
  up "${x#*|}"; echo "== ${x%%|*}$i  XARGS=[${x#*|}]"; (cd engine && .venv/bin/python ../tools/chatbench.py 8080 2 2>&1 | tail -7)
done; done
} > logs/sweep15.txt 2>&1
./strata-rdna3.sh stop >/dev/null 2>&1; ./strata-rdna3.sh serve </dev/null >/dev/null 2>&1
echo "sweep15 done" >> logs/sweep15.txt
