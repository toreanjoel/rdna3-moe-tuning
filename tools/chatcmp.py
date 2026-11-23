# chatcmp.py — how far two arms' greedy answers agree (tools/chatbench.py CHATOUT files): identical answers and the share of each answer before the first differing character
# usage: chatcmp.py A.json B.json
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
same, fr = 0, []
for x, y in zip(a, b):
    n = next((i for i, (c, d) in enumerate(zip(x, y)) if c != d), min(len(x), len(y)))
    same += x == y; fr.append(n / max(1, max(len(x), len(y))))
print(f"{same} of {len(fr)} answers identical; common prefix {sum(fr) / len(fr):.1%} of the answer on average (min {min(fr):.1%})")
