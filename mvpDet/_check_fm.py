import json, pickle, sys

with open("/media/eqip/T9/le/repos_Git/repos_GB/fold0/model_onnx.json") as f:
    j = json.load(f)

with open("/media/eqip/T9/le/repos_Git/repos_GB/fold0/plan_inference.pkl", "rb") as f:
    p = pickle.load(f)

patch_size = j["patch_size"]  # [64, 96, 96] ZYX
fm_json = j["feature_map_size"]

strides = p["architecture"]["strides"]  # [[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,2,2]]
decoder_levels = p["architecture"]["decoder_levels"]  # (2, 3, 4, 5)

print("patch_size:", patch_size)
print("strides:", strides)
print("decoder_levels:", decoder_levels)
print("feature_map_size from JSON:", fm_json)
print()

# Compute cumulative downsampling from encoder strides
# Encoder has len(strides) stages. After stage i, size = size / cumulative_stride[0..i]
import numpy as np
cum = np.array([1, 1, 1])
all_fm = []
for i, s in enumerate(strides):
    cum = cum * np.array(s)
    fm = [int(patch_size[d] / cum[d]) for d in range(3)]
    all_fm.append(fm)
    print("After stride", i, "(cum", cum.tolist(), "):", fm)

print()
print("decoder_levels:", decoder_levels)
# decoder_levels = (2,3,4,5) -> these are 1-indexed or 0-indexed?
# Let's try both and compare with JSON
print()
print("If decoder_levels are 0-indexed (picking levels 2,3,4,5 from all_fm):")
for dl in decoder_levels:
    if dl < len(all_fm):
        print("  level", dl, ":", all_fm[dl])
    else:
        print("  level", dl, ": OUT OF RANGE")

print()
print("If decoder_levels are 1-indexed (picking levels 1,2,3,4 from all_fm):")
for dl in decoder_levels:
    idx = dl - 1
    if 0 <= idx < len(all_fm):
        print("  level", dl, "(idx", idx, "):", all_fm[idx])
    else:
        print("  level", dl, ": OUT OF RANGE")

sys.stdout.flush()
