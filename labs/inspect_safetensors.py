"""Read a .safetensors header without loading any weights and list every tensor.

A .safetensors file is: 8 bytes (little-endian u64 = header length N), then N bytes of JSON
(name -> dtype, shape, byte offsets), then the raw tensor bytes back to back.

usage: python inspect_safetensors.py path/to/model.safetensors [out.json]
"""
import json, math, struct, sys

BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8,
         "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}

path = sys.argv[1]
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
meta = header.pop("__metadata__", {})
rows, total = [], 0
for name, t in header.items():
    count = math.prod(t["shape"])
    total += count
    rows.append({"name": name, "dtype": t["dtype"], "shape": t["shape"], "params": count,
                 "bytes": t["data_offsets"][1] - t["data_offsets"][0]})
rows.sort(key=lambda r: r["name"])
print(f"header: {n} bytes, metadata: {meta}, tensors: {len(rows)}, params: {total:,}")
for r in rows[:40]:
    print(f'{r["name"]:<55} {r["dtype"]:<5} {str(r["shape"]):<14} {r["params"]:>12,}')
if len(sys.argv) > 2:
    json.dump({"header_bytes": n, "metadata": meta, "params": total, "tensors": rows},
              open(sys.argv[2], "w"), indent=1)
