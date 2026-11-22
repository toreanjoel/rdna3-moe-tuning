#!/usr/bin/env python3
# gguf_tensors.py — list a GGUF's tensors (name, ggml type, shape) without the gguf package; filter by substring
# usage: python3 -I tools/gguf_tensors.py model.gguf [substring ...]
import struct, sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
         13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
         22: "IQ2_S", 23: "IQ4_XS", 30: "BF16", 42: "Q2_0"}


def main(path, subs):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    rs = lambda: f.read(rd("Q")).decode("utf-8", "replace")
    assert f.read(4) == b"GGUF"
    rd("I"); nt = rd("Q"); nkv = rd("Q")
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmts = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def skip(t):
        if t == 8: rs()
        elif t == 9:
            et = rd("I"); n = rd("Q")
            if et in sizes: f.seek(sizes[et] * n, 1)
            else:
                for _ in range(n): skip(et)
        else: rd(fmts[t])
    for _ in range(nkv):
        rs(); skip(rd("I"))
    for _ in range(nt):
        name = rs(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; ty = rd("I"); rd("Q")
        if not subs or any(s in name for s in subs):
            print(f"{name:48s} {TYPES.get(ty, ty)!s:8s} {dims}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
