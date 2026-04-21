"""
Binary2Header converter

Use:
    Single: py -3 convert_model.py <model.bin> [output.h]
    Batch: py -3 convert_model.py --all [input_dir] [output_dir]
    Fix stats: py -3 convert_model.py --fix-stats <feature_stats.h> [output.h]

Magics:
    - DT    (magic: 0x44544D4C)
    - kNN   (magic: 0x4B4E4E4C)
    - RR    (magic: 0x52464D4C)
"""

import struct
import sys
import re
import shutil
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

#cfg
CLASS_NAMES=[
    "decaf_tea",
    "tea",
    "decaf_coffee",
    "coffee",
    "ambient",
]

FEATURE_NAMES_BASE=[
    "gas1_norm_step0", "gas1_norm_step1", "gas1_norm_step2", "gas1_norm_step3",
    "gas1_norm_step4", "gas1_norm_step5", "gas1_norm_step6", "gas1_norm_step7",
    "gas1_norm_step8", "gas1_norm_step9",
    "gas2_norm_step0", "gas2_norm_step1", "gas2_norm_step2", "gas2_norm_step3",
    "gas2_norm_step4", "gas2_norm_step5", "gas2_norm_step6", "gas2_norm_step7",
    "gas2_norm_step8", "gas2_norm_step9",
    "cross_ratio_step0", "cross_ratio_step1", "cross_ratio_step2", "cross_ratio_step3",
    "cross_ratio_step4", "cross_ratio_step5", "cross_ratio_step6", "cross_ratio_step7",
    "cross_ratio_step8", "cross_ratio_step9",
    "diff_step0", "diff_step1", "diff_step2", "diff_step3",
    "diff_step4", "diff_step5", "diff_step6", "diff_step7",
    "diff_step8", "diff_step9",
    "gas1_delta_step0", "gas1_delta_step1", "gas1_delta_step2", "gas1_delta_step3",
    "gas1_delta_step4", "gas1_delta_step5", "gas1_delta_step6", "gas1_delta_step7",
    "gas1_delta_step8",
    "gas2_delta_step0", "gas2_delta_step1", "gas2_delta_step2", "gas2_delta_step3",
    "gas2_delta_step4", "gas2_delta_step5", "gas2_delta_step6", "gas2_delta_step7",
    "gas2_delta_step8",
    "slope1_norm", "slope2_norm", "curvature1_norm", "curvature2_norm",
    "auc1_norm", "auc2_norm", "peak_idx1", "peak_idx2",
    "range1", "range2", "late_early_ratio1", "late_early_ratio2",
    "cross_ratio_mean", "cross_ratio_slope", "cross_ratio_var",
]

FEATURE_NAMES_ENV=[
    "env_gas1_raw",
    "env_gas2_raw",
    "env_gas1_baseline",
    "env_temp1_raw",
    "env_hum1_raw",
    "env_gas2_baseline",
    "env_temp2_raw",
    "env_hum2_raw",
    "env_hum2_extra",
]

BASE_FEATURE_COUNT=73

#magic numbers
MAGIC_RF=0x52464D4C
MAGIC_DT=0x44544D4C
MAGIC_KNN=0x4B4E4E4C

#header map
MODEL_FILE_MAP={
    "dt_model.bin": "dt_model_header.h",
    "knn_model.bin": "knn_model_header.h",
    "rf_model.bin": "rf_model_header.h",
}


def build_full_feature_names(total_count):
    names=list(FEATURE_NAMES_BASE)

    for n in FEATURE_NAMES_ENV:
        names.append(n)

    for i in range(10):
        names.append(f"g1g2_ratio_s{i}")
    for i in range(5):
        names.append(f"log_ratio_s{i}")

    pairs=[
        "g2n2×cr1", "g2n2×crSlope", "g1n2×diff1", "slope2×cr0",
        "g2d2×cr2", "g2n3×crMean", "diff5×cr3", "g1n3×g1d6",
        "g1n2×diff5", "g2n4×g2d3"
    ]
    for p in pairs:
        names.append(f"interact_{p}")

    for i in range(9):
        names.append(f"cr_deriv_s{i}")
    for i in range(9):
        names.append(f"delta_ratio_s{i}")
    for i in range(8):
        names.append(f"g2_accel_s{i}")

    names.append("recovery_ratio_g1")
    names.append("recovery_ratio_g2")
    names.append("max_sensor_divergence")
    names.append("cr_early_late_diff")

    names.append("g1_early_late_ratio")
    names.append("g1_mid_late_ratio")
    names.append("g2_early_late_ratio")
    names.append("g2_mid_late_ratio")
    names.append("cross_sensor_early_ratio")
    names.append("cross_sensor_late_ratio")

    names.append("g1_response_shape")
    names.append("g2_response_shape")

    names.append("g2n2_div_hum")
    names.append("g1n2_div_hum")
    names.append("cr0_div_hum")
    names.append("diff1_div_hum")
    names.append("slope2_div_hum")
    names.append("crMean_div_hum")
    names.append("g2n2_div_temp")
    names.append("cr0_div_temp")

    names.append("fisher_coffee_vs_dtea")
    names.append("fisher_dtea_vs_tea")
    names.append("fisher_dcoffee_vs_coffee")
    names.append("fisher_dtea_vs_dcoffee")

    while len(names) < total_count:
        names.append(f"feature_{len(names)}")

    return names[:total_count]


#feature selection state
FS={
    "loaded": False,
    "selected_count": None,
    "original_count": BASE_FEATURE_COUNT,
    "selected_indices": [],
    "selected_medians": [],
    "selected_iqrs": [],
    "selected_means": [],
    "selected_stds": [],
    "uses_robust": False,
}


def extract_array(name, txt):
    i=txt.find(name)
    if i < 0:
        return None
    b1=txt.find("{", i)

    if b1 < 0:
        return None
    b2=txt.find("}", b1)

    if b2 < 0:
        return None
    raw=txt[b1 + 1:b2]
    vals=[]

    for part in raw.split(","):
        v=part.strip()
        if v:
            vals.append(v)

    return vals


def extract_define(name, txt):
    tag="#define " + name
    i=txt.find(tag)
    if i<0:
        tag="#define  " + name
        i=txt.find(tag)

    if i<0:
        return None
    rest=txt[i+len(tag):].lstrip()
    out=""

    for ch in rest:
        if ch.isdigit():
            out+=ch
        else:
            break
    return int(out) if out else None


def load_feature_select(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        txt=f.read()

    FS["selected_count"]=extract_define("SELECTED_FEATURE_COUNT", txt)
    FS["original_count"]=extract_define("ORIGINAL_FEATURE_COUNT", txt) or BASE_FEATURE_COUNT

    idx_vals=extract_array("SELECTED_INDICES", txt)
    FS["selected_indices"]=[]

    if idx_vals:
        for x in idx_vals:
            FS["selected_indices"].append(int(x))

    else:
        print("  warning: could not find SELECTED_INDICES array")

    med=extract_array("SELECTED_MEDIANS", txt)
    iqr=extract_array("SELECTED_IQRS", txt)
    FS["uses_robust"]=False
    FS["selected_medians"]=[]
    FS["selected_iqrs"]=[]
    FS["selected_means"]=[]
    FS["selected_stds"]=[]

    if med and iqr:
        for x in med:
            FS["selected_medians"].append(float(x))

        for x in iqr:
            FS["selected_iqrs"].append(float(x))
        FS["uses_robust"]=True

    else:
        m=extract_array("SELECTED_MEANS", txt)
        s=extract_array("SELECTED_STDS", txt)
        if m:
            for x in m:
                FS["selected_means"].append(float(x))
                
        if s:
            for x in s:
                FS["selected_stds"].append(float(x))

    if FS["selected_count"] and FS["selected_indices"]:
        if len(FS["selected_indices"]) != FS["selected_count"]:
            print(
                f"  warning: SELECTED_FEATURE_COUNT ({FS['selected_count']}) != "
                f"len(SELECTED_INDICES) ({len(FS['selected_indices'])})"
            )
            FS["selected_count"]=len(FS["selected_indices"])

    FS["loaded"]=FS["selected_count"] is not None and len(FS["selected_indices"]) > 0
    if FS["loaded"]:
        s="robust (median/IQR)" if FS["uses_robust"] else "z-score (mean/std)"
        print(f"  loaded feature selection: {FS['selected_count']}/{FS['original_count']} features")
        print(f"  scaling type: {s}")
    else:
        print("  failed to parse feature_select.h!")
        print(f"  selected_count: {FS['selected_count']}")
        print(f"  indices found: {len(FS['selected_indices'])}")

    return FS["loaded"]


def try_load_feature_select(search_dir):
    p=Path(search_dir)/"feature_select.h"

    if p.exists():
        return load_feature_select(str(p))
    
    return False


def get_feature_name(idx):
    if FS["loaded"] and idx < len(FS["selected_indices"]):
        orig_idx=FS["selected_indices"][idx]
        names=build_full_feature_names(FS["original_count"])

        if orig_idx < len(names):
            return names[orig_idx]

        return f"feature_{orig_idx}"
    
    names=build_full_feature_names(max(idx + 1, 200))
    if idx < len(names):
        return names[idx]
    
    return f"feature_{idx}"


#shared header writers
def write_file_header(f, model_name, info, feat_count, num_classes):
    f.write("// Auto-generated C++ header file\n")
    f.write(f"// Model Type: {model_name}\n")
    f.write(f"// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"// Feature count: {feat_count}")
    if FS["loaded"]:
        f.write(f" (selected from {FS['original_count']})")
    f.write("\n")
    scaling="robust (median/IQR)" if FS["uses_robust"] else "z-score"
    f.write(f"// Normalisation: {scaling}\n")
    f.write(f"// Classes: {num_classes} ({', '.join(CLASS_NAMES)})\n")
    if info:
        f.write(f"// Info: {info}\n")
    f.write("\n")


def write_class_names(f, prefix, num_classes):
    f.write("// Class Names\n")
    f.write(f"static const char* {prefix}_CLASS_NAMES[{num_classes}]={{\n")
    for i, name in enumerate(CLASS_NAMES):
        comma="," if i < num_classes - 1 else ""
        f.write(f'    "{name}"{comma}\n')
    f.write("};\n\n")


def write_progmem_macros(f):
    f.write("#ifdef __AVR__\n")
    f.write("  #include <avr/pgmspace.h>\n")
    f.write("#else\n")
    f.write("  #ifndef PROGMEM\n")
    f.write("    #define PROGMEM\n")
    f.write("  #endif\n")
    f.write("  #ifndef pgm_read_byte\n")
    f.write("    #define pgm_read_byte(addr) (*(const uint8_t *)(addr))\n")
    f.write("  #endif\n")
    f.write("  #ifndef pgm_read_word\n")
    f.write("    #define pgm_read_word(addr) (*(const uint16_t *)(addr))\n")
    f.write("  #endif\n")
    f.write("  #ifndef pgm_read_dword\n")
    f.write("    #define pgm_read_dword(addr) (*(const uint32_t *)(addr))\n")
    f.write("  #endif\n")
    f.write("  #ifndef pgm_read_float\n")
    f.write("    #define pgm_read_float(addr) (*(const float *)(addr))\n")
    f.write("  #endif\n")
    f.write("  #ifndef memcpy_P\n")
    f.write("    #define memcpy_P(dest, src, n) memcpy((dest), (src), (n))\n")
    f.write("  #endif\n")
    f.write("#endif\n\n")


#dt
def dt_read_tree(f, ver):
    exists=struct.unpack("<B", f.read(1))[0]
    if not exists:
        return None

    label=struct.unpack("<B", f.read(1))[0]
    feat_idx=struct.unpack("<i", f.read(4))[0]
    th=struct.unpack("<f", f.read(4))[0]

    if ver >= 2:
        maj=struct.unpack("<H", f.read(2))[0]
        total=struct.unpack("<H", f.read(2))[0]
    else:
        maj=1
        total=1

    return {
        "label": label,
        "feature_index": feat_idx,
        "threshold": th,
        "majority_count": maj,
        "total_samples": total,
        "left": dt_read_tree(f, ver),
        "right": dt_read_tree(f, ver),
    }


def dt_flatten_tree(tree):
    if tree is None:
        return []

    flat=[]
    q=deque([tree])
    nodes=[]
    idx_of={}

    while q:
        n=q.popleft()
        i=len(nodes)
        idx_of[id(n)]=i
        nodes.append(n)
        if n["left"]:
            q.append(n["left"])
        if n["right"]:
            q.append(n["right"])

    for n in nodes:
        l=idx_of[id(n["left"])] if n["left"] else -1
        r=idx_of[id(n["right"])] if n["right"] else -1
        flat.append(
            {
                "feature_index": n["feature_index"],
                "label": n["label"],
                "threshold": n["threshold"],
                "majority_count": n["majority_count"],
                "total_samples": n["total_samples"],
                "left_child": l,
                "right_child": r,
            }
        )

    return flat


def load_dt(path):
    with open(path, "rb") as f:
        magic, ver=struct.unpack("<IH", f.read(6))
        if magic != MAGIC_DT:
            raise ValueError(f"Invalid DT magic: {hex(magic)}, expected {hex(MAGIC_DT)}")
        if ver not in (1, 2):
            raise ValueError(f"Unsupported DT version: {ver}")

        feat_count=struct.unpack("<H", f.read(2))[0]
        max_depth=struct.unpack("<B", f.read(1))[0]
        min_samples=struct.unpack("<B", f.read(1))[0]
        tree=dt_read_tree(f, ver)

    if FS["loaded"]:
        fs_count=FS["selected_count"]

        if fs_count and fs_count != feat_count:
            print(f"  note: binary says {feat_count} features, feature_select.h says {fs_count}. using {fs_count}.")
            feat_count=fs_count

    flat_nodes=dt_flatten_tree(tree)

    print(f"Loaded DT model: {len(flat_nodes)} nodes, {feat_count} features, max depth {max_depth}")
    return {
        "feature_count": feat_count,
        "num_classes": len(CLASS_NAMES),
        "max_depth": max_depth,
        "min_samples": min_samples,
        "flat_nodes": flat_nodes,
    }


def print_dt_summary(m):
    print("\n=== Decision Tree Model Summary ===")
    print(f"  Total Nodes:  {len(m['flat_nodes'])}")
    print(f"  Max Depth:    {m['max_depth']}")
    print(f"  Min Samples:  {m['min_samples']}")
    print(f"  Features:     {m['feature_count']}")

    leaf_count=0
    for n in m["flat_nodes"]:
        if n["feature_index"] < 0:
            leaf_count += 1
    internal_count=len(m["flat_nodes"]) - leaf_count

    print(f"  Internal:     {internal_count}")
    print(f"  Leaves:       {leaf_count}")

    use={}
    for n in m["flat_nodes"]:
        fi=n["feature_index"]
        if fi >= 0:
            use[fi]=use.get(fi, 0) + 1

    if use:
        print("\n  Feature Usage (splits):")
        for fi in sorted(use.keys()):
            name=get_feature_name(fi)
            cnt=use[fi]
            bar="â–ˆ" * cnt
            print(f"    [{fi:2d}] {name:30s} {cnt:3d} {bar}")


def write_dt_funcs(f, num_classes):
    f.write("// ===========================================================================\n")
    f.write("// DT Inference Functions\n")
    f.write("// ===========================================================================\n\n")

    f.write("#define DT_READ_NODE(idx) ({ \\\n")
    f.write("    dt_node_t _n; \\\n")
    f.write("    memcpy_P(&_n, &DT_NODES[idx], sizeof(dt_node_t)); \\\n")
    f.write("    _n; })\n\n")

    f.write("static inline uint8_t dt_predict(const float* features) {\n")
    f.write("    uint16_t nodeIdx=0;\n")
    f.write("    dt_node_t node=DT_READ_NODE(nodeIdx);\n")
    f.write("    \n")
    f.write("    while (node.featureIndex >= 0) {\n")
    f.write("        if (features[node.featureIndex] < node.threshold) {\n")
    f.write("            nodeIdx=node.leftChild;\n")
    f.write("        } else {\n")
    f.write("            nodeIdx=node.rightChild;\n")
    f.write("        }\n")
    f.write("        node=DT_READ_NODE(nodeIdx);\n")
    f.write("    }\n")
    f.write("    return node.label;\n")
    f.write("}\n\n")

    f.write("static inline uint8_t dt_predict_with_confidence(const float* features, float* confidence_out) {\n")
    f.write("    uint16_t nodeIdx=0;\n")
    f.write("    dt_node_t node=DT_READ_NODE(nodeIdx);\n")
    f.write("    \n")
    f.write("    while (node.featureIndex >= 0) {\n")
    f.write("        if (features[node.featureIndex] < node.threshold) {\n")
    f.write("            nodeIdx=node.leftChild;\n")
    f.write("        } else {\n")
    f.write("            nodeIdx=node.rightChild;\n")
    f.write("        }\n")
    f.write("        node=DT_READ_NODE(nodeIdx);\n")
    f.write("    }\n")
    f.write("    \n")
    f.write("    if (confidence_out) {\n")
    f.write("        const float total=(float)node.totalSamples;\n")
    f.write("        const float majority=(float)node.majorityCount;\n")
    f.write("        *confidence_out=(total > 0.0f) ? (majority / total) : 1.0f;\n")
    f.write("    }\n")
    f.write("    return node.label;\n")
    f.write("}\n\n")

    f.write("static inline const char* dt_get_class_name(uint8_t idx) {\n")
    f.write(f"    return (idx < {num_classes}) ? DT_CLASS_NAMES[idx] : \"unknown\";\n")
    f.write("}\n\n")


def write_dt_header(m, out_path):
    total_nodes=len(m["flat_nodes"])
    est_size=total_nodes * 16

    with open(out_path, "w") as f:
        write_file_header(f, "Decision Tree", f"Nodes: {total_nodes}", m["feature_count"], m["num_classes"])

        f.write("#ifndef DT_MODEL_DATA_H\n")
        f.write("#define DT_MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <string.h>\n\n")

        write_progmem_macros(f)
        f.write("#if defined(SELECTED_FEATURE_COUNT) && SELECTED_FEATURE_COUNT != ")
        f.write(f"{m['feature_count']}\n")
        f.write('  #error "DT model feature count does not match SELECTED_FEATURE_COUNT"\n')
        f.write("#endif\n\n")

        f.write("// Model Parameters\n")
        f.write(f"#define DT_FEATURE_COUNT {m['feature_count']}\n")
        f.write(f"#define DT_TOTAL_NODES {total_nodes}\n")
        f.write(f"#define DT_NUM_CLASSES {m['num_classes']}\n")
        f.write(f"#define DT_MAX_DEPTH {m['max_depth']}\n")
        f.write("#define DT_HAS_CONFIDENCE 1\n\n")

        write_class_names(f, "DT", m["num_classes"])

        f.write("// Node Structure\n")
        f.write("typedef struct {\n")
        f.write("    int8_t featureIndex;\n")
        f.write("    uint8_t label;\n")
        f.write("    float threshold;\n")
        f.write("    uint16_t majorityCount;\n")
        f.write("    uint16_t totalSamples;\n")
        f.write("    int32_t leftChild;\n")
        f.write("    int32_t rightChild;\n")
        f.write("} dt_node_t;\n\n")

        f.write(f"static const dt_node_t DT_NODES[{total_nodes}] PROGMEM={{\n")
        for i, n in enumerate(m["flat_nodes"]):
            is_leaf=n["feature_index"] < 0
            c=""
            if is_leaf:
                lbl=n["label"]
                name=CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"class_{lbl}"
                c=f"  // LEAF: {name} ({n['majority_count']}/{n['total_samples']})"
            else:
                fname=get_feature_name(n["feature_index"])
                c=f"  // {fname} < {n['threshold']:.4f}?"

            f.write(f"    {{{n['feature_index']}, {n['label']}, ")
            f.write(f"{n['threshold']:.6f}f, ")
            f.write(f"{n['majority_count']}, {n['total_samples']}, ")
            f.write(f"{n['left_child']}, {n['right_child']}}}")
            if i < total_nodes - 1:
                f.write(",")
            f.write(f"{c}\n")
        f.write("};\n\n")

        write_dt_funcs(f, m["num_classes"])

        f.write("#endif // DT_MODEL_DATA_H\n")

    print(f"DT header generated: {out_path}")
    print(f"  Estimated flash: {est_size:,} bytes")


#knn
def load_knn(path):
    with open(path, "rb") as f:
        magic, ver=struct.unpack("<IH", f.read(6))

        if magic != MAGIC_KNN:
            raise ValueError(f"Invalid KNN magic: {hex(magic)}, expected {hex(MAGIC_KNN)}")
        if ver != 1:
            raise ValueError(f"Unsupported KNN version: {ver}")

        k=struct.unpack("<B", f.read(1))[0]
        bin_feat_count=struct.unpack("<H", f.read(2))[0]
        sample_count=struct.unpack("<H", f.read(2))[0]

        samples=[]
        for _ in range(sample_count):
            label=struct.unpack("<B", f.read(1))[0]
            feats=list(struct.unpack(f"<{bin_feat_count}f", f.read(4 * bin_feat_count)))
            samples.append({"label": label, "features": feats})

    feat_count=bin_feat_count
    if FS["loaded"]:
        fs_n=FS["selected_count"]
        if fs_n and fs_n != feat_count:
            print(f"  note: binary says {feat_count} features, feature_select.h says {fs_n}. using {fs_n}.")
            feat_count=fs_n

    for s in samples:
        if len(s["features"]) > feat_count:
            s["features"]=s["features"][:feat_count]

        elif len(s["features"]) < feat_count:
            s["features"].extend([0.0] * (feat_count - len(s["features"])))

    print(f"Loaded KNN model: K={k}, {sample_count} samples, {feat_count} features")
    return {
        "k": k,
        "sample_count": sample_count,
        "samples": samples,
        "feature_count": feat_count,
        "num_classes": len(CLASS_NAMES),
    }


def print_knn_summary(m):
    print("\n=== KNN Model Summary ===")
    print(f"  K:        {m['k']}")
    print(f"  Samples:  {m['sample_count']}")
    print(f"  Features: {m['feature_count']}")

    print("\n  Class Distribution:")
    class_counts=[0] * m["num_classes"]
    for s in m["samples"]:
        if s["label"] < m["num_classes"]:
            class_counts[s["label"]] += 1

    for i, cnt in enumerate(class_counts):
        name=CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        bar="â–ˆ" * cnt
        print(f"    [{i:2d}] {name:30s} {cnt:3d} {bar}")

    flash=m["sample_count"] * (1 + m["feature_count"] * 4)
    stack=m["k"] * 8
    print(f"\n  Flash estimate:       {flash:,} bytes")
    print(f"  Inference stack est:  ~{stack} bytes")
    if flash > 65536:
        print("  WARNING: Large model, may be slow on embedded targets")


def write_knn_funcs(f, feat_count, sample_count, num_classes):
    f.write("// ===========================================================================\n")
    f.write("// KNN Inference Functions\n")
    f.write("// ===========================================================================\n\n")

    f.write("static inline float knn_euclidean_distance(const float* a, const float* b) {\n")
    f.write("    float sum=0.0f;\n")
    f.write(f"    for (uint16_t i=0; i < {feat_count}; i++) {{\n")
    f.write("        float d=a[i] - b[i];\n")
    f.write("        sum += d * d;\n")
    f.write("    }\n")
    f.write("    return sqrtf(sum);\n")
    f.write("}\n\n")

    f.write("static inline void knn_find_neighbors(const float* features, knn_neighbor_t* neighbors) {\n")
    f.write("    for (uint8_t k=0; k < KNN_K; k++) {\n")
    f.write("        neighbors[k].distance=1e30f;\n")
    f.write("        neighbors[k].label=0;\n")
    f.write("    }\n\n")
    f.write(f"    for (uint16_t i=0; i < {sample_count}; i++) {{\n")
    f.write("        knn_sample_t sample;\n")
    f.write("        memcpy_P(&sample, &KNN_SAMPLES[i], sizeof(knn_sample_t));\n")
    f.write("        float dist=knn_euclidean_distance(features, sample.features);\n\n")
    f.write("        if (dist < neighbors[KNN_K - 1].distance) {\n")
    f.write("            neighbors[KNN_K - 1].distance=dist;\n")
    f.write("            neighbors[KNN_K - 1].label=sample.label;\n\n")
    f.write("            // Bubble up to maintain sorted order\n")
    f.write("            for (int8_t j=KNN_K - 2; j >= 0; j--) {\n")
    f.write("                if (neighbors[j + 1].distance < neighbors[j].distance) {\n")
    f.write("                    knn_neighbor_t tmp=neighbors[j];\n")
    f.write("                    neighbors[j]=neighbors[j + 1];\n")
    f.write("                    neighbors[j + 1]=tmp;\n")
    f.write("                } else {\n")
    f.write("                    break;\n")
    f.write("                }\n")
    f.write("            }\n")
    f.write("        }\n")
    f.write("    }\n")
    f.write("}\n\n")

    f.write("static inline uint8_t knn_predict(const float* features) {\n")
    f.write("    knn_neighbor_t neighbors[KNN_K];\n")
    f.write("    knn_find_neighbors(features, neighbors);\n\n")
    f.write(f"    uint16_t votes[{num_classes}]={{0}};\n")
    f.write("    for (uint8_t k=0; k < KNN_K; k++) {\n")
    f.write(f"        if (neighbors[k].label < {num_classes}) {{\n")
    f.write("            votes[neighbors[k].label]++;\n")
    f.write("        }\n")
    f.write("    }\n\n")
    f.write("    uint8_t best=0;\n")
    f.write("    uint16_t maxVotes=0;\n")
    f.write(f"    for (uint8_t i=0; i < {num_classes}; i++) {{\n")
    f.write("        if (votes[i] > maxVotes) {\n")
    f.write("            maxVotes=votes[i];\n")
    f.write("            best=i;\n")
    f.write("        }\n")
    f.write("    }\n")
    f.write("    return best;\n")
    f.write("}\n\n")

    f.write("static inline uint8_t knn_predict_with_confidence(const float* features, float* confidence) {\n")
    f.write("    knn_neighbor_t neighbors[KNN_K];\n")
    f.write("    knn_find_neighbors(features, neighbors);\n\n")
    f.write("    // Distance-weighted voting (matches trainer)\n")
    f.write(f"    float scores[{num_classes}];\n")
    f.write("    memset(scores, 0, sizeof(scores));\n")
    f.write("    float totalWeight=0.0f;\n\n")
    f.write("    for (uint8_t k=0; k < KNN_K; k++) {\n")
    f.write("        float w=1.0f / (neighbors[k].distance + 1e-5f);\n")
    f.write(f"        if (neighbors[k].label < {num_classes}) {{\n")
    f.write("            scores[neighbors[k].label] += w;\n")
    f.write("            totalWeight += w;\n")
    f.write("        }\n")
    f.write("    }\n\n")
    f.write("    uint8_t best=0;\n")
    f.write("    float maxScore=0.0f;\n")
    f.write(f"    for (uint8_t i=0; i < {num_classes}; i++) {{\n")
    f.write("        if (scores[i] > maxScore) {\n")
    f.write("            maxScore=scores[i];\n")
    f.write("            best=i;\n")
    f.write("        }\n")
    f.write("    }\n\n")
    f.write("    *confidence=(totalWeight > 0.0f) ? (maxScore / totalWeight) : 0.0f;\n")
    f.write("    return best;\n")
    f.write("}\n\n")

    f.write("static inline const char* knn_get_class_name(uint8_t idx) {\n")
    f.write(f"    return (idx < {num_classes}) ? KNN_CLASS_NAMES[idx] : \"unknown\";\n")
    f.write("}\n\n")


def write_knn_header(m, out_path):
    est_size=m["sample_count"] * (1 + m["feature_count"] * 4)
    stack=m["k"] * 8

    with open(out_path, "w") as f:
        write_file_header(f, "K-Nearest Neighbors", f"K={m['k']}, Samples={m['sample_count']}", m["feature_count"], m["num_classes"])

        f.write("#ifndef KNN_MODEL_DATA_H\n")
        f.write("#define KNN_MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <string.h>\n")
        f.write("#include <math.h>\n\n")

        write_progmem_macros(f)
        f.write(f"#if defined(SELECTED_FEATURE_COUNT) && SELECTED_FEATURE_COUNT != {m['feature_count']}\n")
        f.write('  #error "KNN model feature count does not match SELECTED_FEATURE_COUNT"\n')
        f.write("#endif\n\n")

        f.write("// Model Parameters\n")
        f.write(f"#define KNN_K {m['k']}\n")
        f.write(f"#define KNN_FEATURE_COUNT {m['feature_count']}\n")
        f.write(f"#define KNN_SAMPLE_COUNT {m['sample_count']}\n")
        f.write(f"#define KNN_NUM_CLASSES {m['num_classes']}\n\n")

        write_class_names(f, "KNN", m["num_classes"])

        f.write("// Sample Structure\n")
        f.write("typedef struct {\n")
        f.write("    uint8_t label;\n")
        f.write(f"    float features[{m['feature_count']}];\n")
        f.write("} knn_sample_t;\n\n")

        f.write("// Neighbor Structure (used during inference)\n")
        f.write("typedef struct {\n")
        f.write("    float distance;\n")
        f.write("    uint8_t label;\n")
        f.write("} knn_neighbor_t;\n\n")

        f.write(f"static const knn_sample_t KNN_SAMPLES[{m['sample_count']}] PROGMEM={{\n")
        for i, s in enumerate(m["samples"]):
            lbl=s["label"]
            name=CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"class_{lbl}"
            f.write(f"    {{{lbl}, {{")
            for j, feat in enumerate(s["features"]):
                f.write(f"{feat:.6f}f")
                if j < len(s["features"]) - 1:
                    f.write(", ")
            f.write("}}")
            if i < m["sample_count"] - 1:
                f.write(",")
            f.write(f"  // {name}\n")
        f.write("};\n\n")

        write_knn_funcs(f, m["feature_count"], m["sample_count"], m["num_classes"])

        f.write("#endif // KNN_MODEL_DATA_H\n")

    print(f"KNN header generated: {out_path}")
    print(f"  Estimated flash: {est_size:,} bytes")
    print(f"  Inference stack: ~{stack} bytes (K={m['k']} neighbors)")


#rf
def rf_read_tree(f):
    if not struct.unpack("<B", f.read(1))[0]:
        return None

    label=struct.unpack("<B", f.read(1))[0]
    feat_idx=struct.unpack("<i", f.read(4))[0]
    th=struct.unpack("<f", f.read(4))[0]

    return {
        "label": label,
        "feature_index": feat_idx,
        "threshold": th,
        "left": rf_read_tree(f),
        "right": rf_read_tree(f),
    }


def rf_flatten_trees(trees):
    flat=[]
    roots=[]

    for tree in trees:
        roots.append(len(flat))
        if tree is None:
            continue

        q=deque([tree])
        nodes=[]
        idx_map={}

        while q:
            n=q.popleft()
            i=len(nodes)
            idx_map[id(n)]=i
            nodes.append(n)
            if n["left"]:
                q.append(n["left"])
            if n["right"]:
                q.append(n["right"])

        off=len(flat)
        for n in nodes:
            l=-1
            r=-1
            if n["left"]:
                l=off + idx_map[id(n["left"])]
            if n["right"]:
                r=off + idx_map[id(n["right"])]
            flat.append(
                {
                    "feature_index": n["feature_index"],
                    "label": n["label"],
                    "threshold": n["threshold"],
                    "left_child": l,
                    "right_child": r,
                }
            )

    return flat, roots


def load_rf(path):
    with open(path, "rb") as f:
        magic, ver=struct.unpack("<IH", f.read(6))

        if magic != MAGIC_RF:
            raise ValueError(f"Invalid RF magic: {hex(magic)}, expected {hex(MAGIC_RF)}")
        if ver != 1:
            raise ValueError(f"Unsupported RF version: {ver}")

        num_trees=struct.unpack("<H", f.read(2))[0]
        max_depth=struct.unpack("<B", f.read(1))[0]
        min_samples=struct.unpack("<B", f.read(1))[0]
        feat_ratio=struct.unpack("<f", f.read(4))[0]
        bin_feat_count=struct.unpack("<H", f.read(2))[0]
        feat_subset_size=struct.unpack("<H", f.read(2))[0]
        oob=struct.unpack("<f", f.read(4))[0]

        feat_imp=[]
        for _ in range(bin_feat_count):
            feat_imp.append(struct.unpack("<f", f.read(4))[0])

        trees=[]
        tree_count=struct.unpack("<H", f.read(2))[0]
        for _ in range(tree_count):
            trees.append(rf_read_tree(f))

    feat_count=bin_feat_count
    if FS["loaded"]:
        chosen=FS["selected_count"]
        if chosen and chosen != feat_count:
            print(f"  note: binary says {feat_count} features, feature_select.h says {chosen}. using {chosen}.")
            feat_count=chosen

    if len(feat_imp) > feat_count:
        feat_imp=feat_imp[:feat_count]
    elif len(feat_imp) < feat_count:
        feat_imp.extend([0.0] * (feat_count - len(feat_imp)))

    flat_nodes, roots=rf_flatten_trees(trees)
    print(
        f"Loaded RF model: {len(trees)} trees, {len(flat_nodes)} total nodes, "
        f"{feat_count} features, OOB error {oob * 100:.2f}%"
    )

    return {
        "num_classes": len(CLASS_NAMES),
        "num_trees": num_trees,
        "max_depth": max_depth,
        "min_samples": min_samples,
        "feature_subset_ratio": feat_ratio,
        "feature_subset_size": feat_subset_size,
        "oob_error": oob,
        "feature_count": feat_count,
        "feature_importance": feat_imp,
        "trees": trees,
        "flat_nodes": flat_nodes,
        "tree_roots": roots,
    }


def print_rf_summary(m):
    print("\n=== Random Forest Model Summary ===")
    print(f"  Trees:       {len(m['trees'])}")
    print(f"  Total Nodes: {len(m['flat_nodes'])}")
    print(f"  Max Depth:   {m['max_depth']}")
    print(f"  Min Samples: {m['min_samples']}")
    print(f"  Features:    {m['feature_count']}")
    print(f"  Subset Size: {m['feature_subset_size']}")
    print(f"  Subset Ratio:{m['feature_subset_ratio']:.2f}")
    print(f"  OOB Error:   {m['oob_error'] * 100:.2f}%")

    node_counts=[]
    for i in range(len(m["tree_roots"])):
        start=m["tree_roots"][i]
        end=m["tree_roots"][i + 1] if i + 1 < len(m["tree_roots"]) else len(m["flat_nodes"])
        node_counts.append(end - start)

    if node_counts:
        avg_nodes=sum(node_counts) / len(node_counts)
        print(f"  Avg Nodes/Tree: {avg_nodes:.1f}")
        print(f"  Min Nodes/Tree: {min(node_counts)}")
        print(f"  Max Nodes/Tree: {max(node_counts)}")

    if m["feature_importance"]:
        print("\n  Feature Importance:")
        pairs=sorted(enumerate(m["feature_importance"]), key=lambda x: x[1], reverse=True)
        max_imp=max(m["feature_importance"]) if m["feature_importance"] else 1.0
        for idx, imp in pairs:
            name=get_feature_name(idx)
            bar_len=int((imp / max_imp) * 30) if max_imp > 0 else 0
            bar="â–ˆ" * bar_len
            print(f"    [{idx:2d}] {name:35s} {imp:.6f} {bar}")

    est_flash=len(m["flat_nodes"]) * 16
    print(f"\n  Flash estimate: {est_flash:,} bytes")


def write_rf_funcs(f, num_classes):
    f.write("// ===========================================================================\n")
    f.write("// RF Inference Functions\n")
    f.write("// ===========================================================================\n\n")

    f.write("#define RF_READ_NODE(idx) ({ \\\n")
    f.write("    rf_node_t _n; \\\n")
    f.write("    memcpy_P(&_n, &RF_NODES[idx], sizeof(rf_node_t)); \\\n")
    f.write("    _n; })\n\n")

    f.write("static inline uint8_t rf_predict_tree(uint32_t treeIdx, const float* features) {\n")
    f.write("    uint32_t nodeIdx=pgm_read_dword(&RF_TREE_ROOTS[treeIdx]);\n")
    f.write("    rf_node_t node=RF_READ_NODE(nodeIdx);\n")
    f.write("    \n")
    f.write("    while (node.featureIndex >= 0) {\n")
    f.write("        if (features[node.featureIndex] < node.threshold) {\n")
    f.write("            nodeIdx=node.leftChild;\n")
    f.write("        } else {\n")
    f.write("            nodeIdx=node.rightChild;\n")
    f.write("        }\n")
    f.write("        node=RF_READ_NODE(nodeIdx);\n")
    f.write("    }\n")
    f.write("    return node.label;\n")
    f.write("}\n\n")

    f.write("static inline uint8_t rf_predict(const float* features) {\n")
    f.write(f"    uint16_t votes[{num_classes}]={{0}};\n")
    f.write("    for (uint32_t t=0; t < RF_NUM_TREES; t++) {\n")
    f.write("        uint8_t pred=rf_predict_tree(t, features);\n")
    f.write(f"        if (pred < {num_classes}) votes[pred]++;\n")
    f.write("    }\n\n")
    f.write("    uint8_t best=0;\n")
    f.write("    uint16_t maxVotes=0;\n")
    f.write(f"    for (uint8_t i=0; i < {num_classes}; i++) {{\n")
    f.write("        if (votes[i] > maxVotes) {\n")
    f.write("            maxVotes=votes[i];\n")
    f.write("            best=i;\n")
    f.write("        }\n")
    f.write("    }\n")
    f.write("    return best;\n")
    f.write("}\n\n")

    f.write("static inline uint8_t rf_predict_with_confidence(const float* features, float* confidence) {\n")
    f.write(f"    uint16_t votes[{num_classes}]={{0}};\n")
    f.write("    for (uint32_t t=0; t < RF_NUM_TREES; t++) {\n")
    f.write("        uint8_t pred=rf_predict_tree(t, features);\n")
    f.write(f"        if (pred < {num_classes}) votes[pred]++;\n")
    f.write("    }\n\n")
    f.write("    uint8_t best=0;\n")
    f.write("    uint16_t maxVotes=0;\n")
    f.write(f"    for (uint8_t i=0; i < {num_classes}; i++) {{\n")
    f.write("        if (votes[i] > maxVotes) {\n")
    f.write("            maxVotes=votes[i];\n")
    f.write("            best=i;\n")
    f.write("        }\n")
    f.write("    }\n\n")
    f.write("    *confidence=(float)maxVotes / (float)RF_NUM_TREES;\n")
    f.write("    return best;\n")
    f.write("}\n\n")

    f.write("static inline const char* rf_get_class_name(uint8_t idx) {\n")
    f.write(f"    return (idx < {num_classes}) ? RF_CLASS_NAMES[idx] : \"unknown\";\n")
    f.write("}\n\n")


def write_rf_header(m, out_path):
    total_nodes=len(m["flat_nodes"])
    est_size=total_nodes * 16

    with open(out_path, "w") as f:
        write_file_header(
            f,
            "Random Forest",
            f"Trees: {len(m['trees'])}, Nodes: {total_nodes}, OOB Error: {m['oob_error'] * 100:.2f}%",
            m["feature_count"],
            m["num_classes"],
        )

        f.write("#ifndef RF_MODEL_DATA_H\n")
        f.write("#define RF_MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <string.h>\n\n")

        write_progmem_macros(f)
        f.write(f"#if defined(SELECTED_FEATURE_COUNT) && SELECTED_FEATURE_COUNT != {m['feature_count']}\n")
        f.write('  #error "RF model feature count does not match SELECTED_FEATURE_COUNT"\n')
        f.write("#endif\n\n")

        f.write("// Model Parameters\n")
        f.write(f"#define RF_NUM_TREES {len(m['trees'])}\n")
        f.write(f"#define RF_MAX_DEPTH {m['max_depth']}\n")
        f.write(f"#define RF_FEATURE_COUNT {m['feature_count']}\n")
        f.write(f"#define RF_TOTAL_NODES {total_nodes}\n")
        f.write(f"#define RF_NUM_CLASSES {m['num_classes']}\n")
        f.write(f"#define RF_OOB_ERROR {m['oob_error']:.6f}f\n\n")

        write_class_names(f, "RF", m["num_classes"])

        f.write("// Feature Importance (normalized)\n")
        f.write(f"static const float RF_FEATURE_IMPORTANCE[{m['feature_count']}] PROGMEM={{\n")
        for i, imp in enumerate(m["feature_importance"]):
            name=get_feature_name(i)
            f.write(f"    {imp:.6f}f")
            if i < m["feature_count"] - 1:
                f.write(",")
            f.write(f"  // [{i}] {name}\n")
        f.write("};\n\n")

        f.write("// Node Structure\n")
        f.write("typedef struct {\n")
        f.write("    int8_t featureIndex;\n")
        f.write("    uint8_t label;\n")
        f.write("    float threshold;\n")
        f.write("    int32_t leftChild;\n")
        f.write("    int32_t rightChild;\n")
        f.write("} rf_node_t;\n\n")
        f.write(f"static const rf_node_t RF_NODES[{total_nodes}] PROGMEM={{\n")

        for i, n in enumerate(m["flat_nodes"]):
            if i in m["tree_roots"]:
                tree_idx=m["tree_roots"].index(i)
                f.write(f"    // --- Tree {tree_idx} ---\n")

            f.write(f"    {{{n['feature_index']}, {n['label']}, ")
            f.write(f"{n['threshold']:.6f}f, ")
            f.write(f"{n['left_child']}, {n['right_child']}}}")
            if i < total_nodes - 1:
                f.write(",")
            f.write("\n")

        f.write("};\n\n")

        f.write(f"static const uint32_t RF_TREE_ROOTS[{len(m['trees'])}] PROGMEM={{")
        for i, off in enumerate(m["tree_roots"]):
            if i > 0:
                f.write(", ")
            if i % 10 == 0:
                f.write("\n    ")
            f.write(str(off))
        f.write("\n};\n\n")

        write_rf_funcs(f, m["num_classes"])

        f.write("#endif // RF_MODEL_DATA_H\n")

    print(f"RF header generated: {out_path}")
    print(f"  Estimated flash: {est_size:,} bytes")


#feature stats converter
def convert_feature_stats(in_path, out_path):
    with open(in_path, "r", encoding="utf-8-sig") as f:
        txt=f.read()

    uses_robust=False
    raw_medians=extract_array("FEATURE_MEDIANS", txt)
    raw_iqrs=extract_array("FEATURE_IQRS", txt)

    if raw_medians and raw_iqrs:
        uses_robust=True
        center_vals=[]
        for x in raw_medians:
            center_vals.append(float(x))
        scale_vals=[]
        for x in raw_iqrs:
            scale_vals.append(float(x))
        center_name="FEATURE_MEDIANS"
        scale_name="FEATURE_IQRS"
        center_comment="medians"
        scale_comment="inter-quartile ranges"
    else:
        raw_means=extract_array("FEATURE_MEANS", txt)
        raw_stds=extract_array("FEATURE_STDS", txt)
        if not raw_means:
            raw_means=extract_array("FEATURE_MEAN", txt)
        if not raw_stds:
            raw_stds=extract_array("FEATURE_STD", txt)

        if not raw_means or not raw_stds:
            raise ValueError(
                "Could not parse feature stats. "
                "Expected FEATURE_MEDIANS/FEATURE_IQRS or FEATURE_MEANS/FEATURE_STDS"
            )

        center_vals=[]
        for x in raw_means:
            center_vals.append(float(x))
        scale_vals=[]
        for x in raw_stds:
            scale_vals.append(float(x))
        center_name="FEATURE_MEANS"
        scale_name="FEATURE_STDS"
        center_comment="means"
        scale_comment="standard deviations"

    count=len(center_vals)
    if len(scale_vals) != count:
        raise ValueError(f"Center count ({len(center_vals)}) != scale count ({len(scale_vals)})")

    fc_match=re.search(r"#define\s+FULL_FEATURE_COUNT\s+(\d+)", txt)
    declared_count=int(fc_match.group(1)) if fc_match else count
    if declared_count != count:
        print(f"  WARNING: FULL_FEATURE_COUNT={declared_count} but found {count} values.")

    names=build_full_feature_names(count)
    scaling="robust (median/IQR)" if uses_robust else "z-score (mean/std)"

    with open(out_path, "w") as f:
        f.write("// Auto-generated feature statistics for Normalisation\n")
        f.write(f"// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"// Full feature space: {count} features\n")
        f.write(f"// Scaling type: {scaling}\n")
        if uses_robust:
            f.write("// Robust scaling: (x - median) / IQR\n")
        else:
            f.write("// Z-score scaling: (x - mean) / std\n")
        f.write("// NOTE: On Arduino, use SELECTED_MEDIANS/SELECTED_IQRS from feature_select.h\n\n")

        f.write("#ifndef FEATURE_STATS_H\n")
        f.write("#define FEATURE_STATS_H\n\n")
        f.write(f"#define FULL_FEATURE_COUNT {count}\n")
        f.write(f"#define FEATURE_SCALING_ROBUST {1 if uses_robust else 0}\n\n")

        f.write(f"// Feature {center_comment}\n")
        f.write(f"static const float {center_name}[{count}]={{\n")
        for i, v in enumerate(center_vals):
            name=names[i] if i < len(names) else f"feature_{i}"
            comma="," if i < count - 1 else ""
            f.write(f"    {v:.8f}f{comma}  // [{i}] {name}\n")
        f.write("};\n\n")

        f.write(f"// Feature {scale_comment}\n")
        f.write(f"static const float {scale_name}[{count}]={{\n")
        for i, v in enumerate(scale_vals):
            name=names[i] if i < len(names) else f"feature_{i}"
            comma="," if i < count - 1 else ""
            f.write(f"    {v:.8f}f{comma}  // [{i}] {name}\n")
        f.write("};\n\n")

        f.write("static inline float normalize_feature(float value, int idx) {\n")
        f.write(f"    if (idx < 0 || idx >= {count}) return value;\n")
        if uses_robust:
            f.write(f"    return (value - {center_name}[idx]) / {scale_name}[idx];\n")
        else:
            f.write(f"    float s={scale_name}[idx];\n")
            f.write("    if (s < 1e-6f) return 0.0f;\n")
            f.write(f"    return (value - {center_name}[idx]) / s;\n")
        f.write("}\n\n")
        f.write("#endif // FEATURE_STATS_H\n")

    print(f"Feature stats header generated: {out_path}")
    print(f"  Features: {count}, Scaling: {scaling}")


#feature select converter
def convert_feature_select(in_path, out_path):
    local={
        "selected_count": None,
        "original_count": BASE_FEATURE_COUNT,
        "selected_indices": [],
        "selected_medians": [],
        "selected_iqrs": [],
        "selected_means": [],
        "selected_stds": [],
        "uses_robust": False,
    }

    with open(in_path, "r", encoding="utf-8-sig") as f:
        txt=f.read()

    local["selected_count"]=extract_define("SELECTED_FEATURE_COUNT", txt)
    local["original_count"]=extract_define("ORIGINAL_FEATURE_COUNT", txt) or BASE_FEATURE_COUNT

    idx_vals=extract_array("SELECTED_INDICES", txt)
    if idx_vals:
        for x in idx_vals:
            local["selected_indices"].append(int(x))

    med=extract_array("SELECTED_MEDIANS", txt)
    iqr=extract_array("SELECTED_IQRS", txt)
    if med and iqr:
        local["uses_robust"]=True
        for x in med:
            local["selected_medians"].append(float(x))
        for x in iqr:
            local["selected_iqrs"].append(float(x))
    else:
        m=extract_array("SELECTED_MEANS", txt)
        s=extract_array("SELECTED_STDS", txt)
        if m:
            for x in m:
                local["selected_means"].append(float(x))
        if s:
            for x in s:
                local["selected_stds"].append(float(x))

    if not (local["selected_count"] and local["selected_indices"]):
        raise ValueError("Could not parse feature selection from input file")

    if len(local["selected_indices"]) != local["selected_count"]:
        local["selected_count"]=len(local["selected_indices"])

    names=build_full_feature_names(local["original_count"])

    with open(out_path, "w") as f:
        scaling="robust (median/IQR)" if local["uses_robust"] else "z-score (mean/std)"

        f.write("// Auto-generated feature selection header\n")
        f.write(f"// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"// Selected {local['selected_count']} features from {local['original_count']}\n")
        f.write(f"// Scaling type: {scaling}\n")
        f.write("// Used by ml_inference.cpp for feature projection and Normalisation\n\n")

        f.write("#ifndef FEATURE_SELECT_H\n")
        f.write("#define FEATURE_SELECT_H\n\n")

        f.write(f"#define SELECTED_FEATURE_COUNT {local['selected_count']}\n")
        f.write(f"#define ORIGINAL_FEATURE_COUNT {local['original_count']}\n")
        f.write(f"#define FEATURE_SCALING_ROBUST {1 if local['uses_robust'] else 0}\n\n")

        f.write(f"static const int SELECTED_INDICES[{local['selected_count']}]={{\n")
        for i, idx in enumerate(local["selected_indices"]):
            name=names[idx] if idx < len(names) else f"feature_{idx}"
            comma="," if i < local["selected_count"] - 1 else ""
            f.write(f"    {idx}{comma}  // [{i}] {name}\n")
        f.write("};\n\n")

        if local["uses_robust"]:
            if local["selected_medians"]:
                f.write(f"static const float SELECTED_MEDIANS[{local['selected_count']}]={{\n")
                for i, m in enumerate(local["selected_medians"]):
                    orig_idx=local["selected_indices"][i] if i < len(local["selected_indices"]) else -1
                    name=names[orig_idx] if 0 <= orig_idx < len(names) else f"feature_{orig_idx}"
                    comma="," if i < local["selected_count"] - 1 else ""
                    f.write(f"    {m:.8f}f{comma}  // [{i}] {name}\n")
                f.write("};\n\n")

            if local["selected_iqrs"]:
                f.write(f"static const float SELECTED_IQRS[{local['selected_count']}]={{\n")
                for i, iqr_val in enumerate(local["selected_iqrs"]):
                    orig_idx=local["selected_indices"][i] if i < len(local["selected_indices"]) else -1
                    name=names[orig_idx] if 0 <= orig_idx < len(names) else f"feature_{orig_idx}"
                    comma="," if i < local["selected_count"] - 1 else ""
                    warning=""
                    if abs(iqr_val - 1.0) < 1e-6:
                        warning="  // NOTE: defaulted IQR"
                    f.write(f"    {iqr_val:.8f}f{comma}  // [{i}] {name}{warning}\n")
                f.write("};\n\n")

            f.write("// Normalize a selected feature using robust scaling\n")
            f.write("static inline float normalize_selected_feature(float value, int selected_idx) {\n")
            f.write(f"    if (selected_idx < 0 || selected_idx >= {local['selected_count']}) return value;\n")
            f.write("    return (value - SELECTED_MEDIANS[selected_idx]) / SELECTED_IQRS[selected_idx];\n")
            f.write("}\n\n")
        else:
            if local["selected_means"]:
                f.write(f"static const float SELECTED_MEANS[{local['selected_count']}]={{\n")
                for i, m in enumerate(local["selected_means"]):
                    orig_idx=local["selected_indices"][i] if i < len(local["selected_indices"]) else -1
                    name=names[orig_idx] if 0 <= orig_idx < len(names) else f"feature_{orig_idx}"
                    comma="," if i < local["selected_count"] - 1 else ""
                    f.write(f"    {m:.8f}f{comma}  // [{i}] {name}\n")
                f.write("};\n\n")

            if local["selected_stds"]:
                f.write(f"static const float SELECTED_STDS[{local['selected_count']}]={{\n")
                for i, s in enumerate(local["selected_stds"]):
                    orig_idx=local["selected_indices"][i] if i < len(local["selected_indices"]) else -1
                    name=names[orig_idx] if 0 <= orig_idx < len(names) else f"feature_{orig_idx}"
                    comma="," if i < local["selected_count"] - 1 else ""
                    f.write(f"    {s:.8f}f{comma}  // [{i}] {name}\n")
                f.write("};\n\n")

            f.write("// Normalize a selected feature using z-score\n")
            f.write("static inline float normalize_selected_feature(float value, int selected_idx) {\n")
            f.write(f"    if (selected_idx < 0 || selected_idx >= {local['selected_count']}) return value;\n")
            f.write("    float std=SELECTED_STDS[selected_idx];\n")
            f.write("    if (std < 1e-6f) return 0.0f;\n")
            f.write("    return (value - SELECTED_MEANS[selected_idx]) / std;\n")
            f.write("}\n\n")

        f.write("#endif // FEATURE_SELECT_H\n")

    print(f"Feature select header generated: {out_path}")
    print(f"  Selected: {local['selected_count']}/{local['original_count']} features")
    print(f"  Scaling:  {'robust (median/IQR)' if local['uses_robust'] else 'z-score (mean/std)'}")


#ensemble weights converter
def convert_ensemble_weights(in_path, out_path):
    with open(in_path, "r") as f:
        txt=f.read()

    dt_match=re.search(r"DT_WEIGHT\s*=\s*([0-9.e+-]+)f?", txt)
    knn_match=re.search(r"KNN_WEIGHT\s*=\s*([0-9.e+-]+)f?", txt)
    rf_match=re.search(r"RF_WEIGHT\s*=\s*([0-9.e+-]+)f?", txt)
    hier_match=re.search(r"HIER_WEIGHT\s*=\s*([0-9.e+-]+)f?", txt)

    if not (dt_match and knn_match and rf_match):
        raise ValueError(
            "Could not parse ensemble weights from input file. "
            "Expected at minimum DT_WEIGHT, KNN_WEIGHT, RF_WEIGHT"
        )

    dt_w=float(dt_match.group(1))
    knn_w=float(knn_match.group(1))
    rf_w=float(rf_match.group(1))
    hier_w=float(hier_match.group(1)) if hier_match else 0.0

    has_hier=hier_match is not None
    weight_power=4.0
    dt_acc=dt_w ** (1.0 / weight_power) if dt_w > 0 else 0.0
    knn_acc=knn_w ** (1.0 / weight_power) if knn_w > 0 else 0.0
    rf_acc=rf_w ** (1.0 / weight_power) if rf_w > 0 else 0.0
    hier_acc=hier_w ** (1.0 / weight_power) if hier_w > 0 else 0.0

    with open(out_path, "w") as f:
        f.write("// Auto-generated ensemble weights\n")
        f.write(f"// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"// Weights are accuracy^{int(weight_power)} from training evaluation\n")
        f.write(f"// DT accuracy:   ~{dt_acc * 100:.1f}%\n")
        f.write(f"// KNN accuracy:  ~{knn_acc * 100:.1f}%\n")
        f.write(f"// RF accuracy:   ~{rf_acc * 100:.1f}%\n")
        if has_hier:
            f.write(f"// Hier accuracy: ~{hier_acc * 100:.1f}%\n")
        f.write("//\n")
        f.write("// Ensemble uses confidence-weighted voting:\n")
        f.write("//   KNN and RF contribute weight * confidence\n")
        f.write("//   Hierarchical uses soft probability routing\n")
        f.write("//   DT is heavily damped (no confidence, weak accuracy)\n\n")

        f.write("#ifndef ENSEMBLE_WEIGHTS_H\n")
        f.write("#define ENSEMBLE_WEIGHTS_H\n\n")

        f.write(f"#define ENSEMBLE_WEIGHT_POWER {int(weight_power)}\n")
        f.write(f"#define ENSEMBLE_HAS_HIERARCHICAL {1 if has_hier else 0}\n\n")

        f.write(f"static const float DT_WEIGHT={dt_w:.6f}f;   // acc ~{dt_acc * 100:.1f}%\n")
        f.write(f"static const float KNN_WEIGHT={knn_w:.6f}f;  // acc ~{knn_acc * 100:.1f}%\n")
        f.write(f"static const float RF_WEIGHT={rf_w:.6f}f;   // acc ~{rf_acc * 100:.1f}%\n")
        if has_hier:
            f.write(f"static const float HIER_WEIGHT={hier_w:.6f}f; // acc ~{hier_acc * 100:.1f}%\n")

        total_w=dt_w + knn_w + rf_w + hier_w
        if total_w > 0:
            f.write("\n// Relative contribution (normalized):\n")
            f.write(f"//   DT:   {dt_w / total_w * 100:.1f}%\n")
            f.write(f"//   KNN:  {knn_w / total_w * 100:.1f}%\n")
            f.write(f"//   RF:   {rf_w / total_w * 100:.1f}%\n")
            if has_hier:
                f.write(f"//   Hier: {hier_w / total_w * 100:.1f}%\n")

        f.write("\n#endif // ENSEMBLE_WEIGHTS_H\n")

    print(f"Ensemble weights header generated: {out_path}")
    print(f"  DT:   {dt_w:.6f} (acc ~{dt_acc * 100:.1f}%)")
    print(f"  KNN:  {knn_w:.6f} (acc ~{knn_acc * 100:.1f}%)")
    print(f"  RF:   {rf_w:.6f} (acc ~{rf_acc * 100:.1f}%)")
    if has_hier:
        print(f"  Hier: {hier_w:.6f} (acc ~{hier_acc * 100:.1f}%)")
    if total_w > 0:
        print(f"  Total weight: {total_w:.6f}")


#detect + convert
def detect_model_type(path):
    with open(path, "rb") as f:
        magic=struct.unpack("<I", f.read(4))[0]

    if magic == MAGIC_DT:
        return "dt"
    if magic == MAGIC_KNN:
        return "knn"
    if magic == MAGIC_RF:
        return "rf"
    raise ValueError(f"Unknown model type. Magic: {hex(magic)}")


def convert_one_model(in_path, out_path):
    mtype=detect_model_type(in_path)
    print(f"Detected model type: {mtype.upper()}")

    if mtype == "dt":
        model=load_dt(in_path)
        print_dt_summary(model)
        write_dt_header(model, out_path)
    elif mtype == "knn":
        model=load_knn(in_path)
        print_knn_summary(model)
        write_knn_header(model, out_path)
    else:
        model=load_rf(in_path)
        print_rf_summary(model)
        write_rf_header(model, out_path)


def convert_all(input_dir=".", output_dir="."):
    in_dir=Path(input_dir)
    out_dir=Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    converted=0
    failed=0

    fs_path=in_dir / "feature_select.h"
    if fs_path.exists():
        try:
            load_feature_select(str(fs_path))
        except Exception as e:
            print(f"warning: could not load feature_select.h: {e}")
    else:
        print(f"note: feature_select.h not found in {in_dir}, will use feature counts from binary files")

    for bin_file, header_file in MODEL_FILE_MAP.items():
        in_path=in_dir / bin_file
        out_path=out_dir / header_file

        if not in_path.exists():
            print(f"Skipping {bin_file} (not found)")
            continue

        try:
            print(f"\n{'=' * 60}")
            print(f"Converting: {bin_file} -> {header_file}")
            print(f"{'=' * 60}")
            convert_one_model(str(in_path), str(out_path))
            converted += 1
        except Exception as e:
            print(f"  FAILED to convert {bin_file}: {e}")
            traceback.print_exc()
            failed += 1

    if fs_path.exists():
        try:
            print(f"\n{'=' * 60}")
            print("Converting: feature_select.h")
            print(f"{'=' * 60}")
            convert_feature_select(str(fs_path), str(out_dir / "feature_select.h"))
            converted += 1
        except Exception as e:
            print(f"FAILED to convert feature_select.h: {e}")
            traceback.print_exc()
            failed += 1

    stats_in=in_dir / "feature_stats.h"
    if stats_in.exists():
        try:
            print(f"\n{'=' * 60}")
            print("Converting: feature_stats.h")
            print(f"{'=' * 60}")
            convert_feature_stats(str(stats_in), str(out_dir / "feature_stats.h"))
            converted += 1
        except Exception as e:
            print(f"FAILED to convert feature_stats.h: {e}")
            traceback.print_exc()
            failed += 1
    else:
        print(f"\nSkipping feature_stats.h (not found in {in_dir})")

    ew_in=in_dir / "ensemble_weights.h"
    if ew_in.exists():
        try:
            print(f"\n{'=' * 60}")
            print("Converting: ensemble_weights.h")
            print(f"{'=' * 60}")
            convert_ensemble_weights(str(ew_in), str(out_dir / "ensemble_weights.h"))
            converted += 1
        except Exception as e:
            print(f"  FAILED to convert ensemble_weights.h: {e}")
            traceback.print_exc()
            failed += 1
    else:
        print(f"\nSkipping ensemble_weights.h (not found in {in_dir})")

    at_in=in_dir / "anomaly_threshold.h"
    if at_in.exists():
        try:
            print(f"\n{'=' * 60}")
            print("Validating: anomaly_threshold.h")
            print(f"{'=' * 60}")

            at_out=out_dir / "anomaly_threshold.h"
            if at_in.resolve() != at_out.resolve():
                shutil.copy2(str(at_in), str(at_out))
                print(f"  Copied anomaly_threshold.h to {out_dir}")
            else:
                print("  Input and output are same file, skipping copy")

            with open(str(at_in), "r") as atf:
                at_txt=atf.read()

            th_match=re.search(r"CALIBRATED_ANOMALY_THRESHOLD\s*=\s*([0-9.e+-]+)f?", at_txt)
            sc_match=re.search(r"KNN_DISTANCE_SCALE\s*=\s*([0-9.e+-]+)f?", at_txt)

            if th_match and sc_match:
                th=float(th_match.group(1))
                sc=float(sc_match.group(1))
                print(f"  Anomaly threshold: {th:.6f}")
                print(f"  KNN distance scale: {sc:.6f}")
                if th == 0.0 or sc == 0.0:
                    print("  WARNING: Zero values detected! Calibration may have failed.")
            else:
                print("  WARNING: Could not parse threshold values")

            converted += 1
        except Exception as e:
            print(f"  FAILED to process anomaly_threshold.h: {e}")
            failed += 1

    total_possible=len(MODEL_FILE_MAP) + 4
    total_attempted=converted + failed
    skipped=total_possible - total_attempted

    print(f"\n{'=' * 60}")
    print("Batch conversion complete")
    print(f"  Converted: {converted}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {skipped}")

    if FS["loaded"]:
        scaling="robust (median/IQR)" if FS["uses_robust"] else "z-score (mean/std)"
        print(f"\n  Feature space: {FS['selected_count']} selected from {FS['original_count']}")
        print(f"  Scaling: {scaling}")
    print(f"  Classes: {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
    print(f"{'=' * 60}")

    return converted


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nSupported magic numbers:")
        print(f"  Decision Tree:  {hex(MAGIC_DT)}")
        print(f"  KNN:            {hex(MAGIC_KNN)}")
        print(f"  Random Forest:  {hex(MAGIC_RF)}")
        print()
        print(f"Classes ({len(CLASS_NAMES)}):")
        
        for i, name in enumerate(CLASS_NAMES):
            print(f"  [{i}] {name}")

        print()
        print(f"Base feature space: {BASE_FEATURE_COUNT} features (73 base + 9 env=82)")
        print("Engineered features: up to ~90 additional")
        print("Feature selection: loaded from feature_select.h if present")
        print("Normalisation: robust scaling (median/IQR) or z-score (mean/std)")
        print()

        print("Examples:")
        print("  python convert_model_rewrite.py rf_model.bin rf_model_header.h")
        print("  python convert_model_rewrite.py --all ./build ./src/model_headers")
        print("  python convert_model_rewrite.py --fix-stats feature_stats.h feature_stats_fixed.h")
        print("  python convert_model_rewrite.py --fix-select feature_select.h feature_select_fixed.h")
        print("  python convert_model_rewrite.py --fix-ensemble ensemble_weights.h ensemble_weights_fixed.h")
        sys.exit(1)

    if sys.argv[1] == "--all":
        in_dir=sys.argv[2] if len(sys.argv) > 2 else "."
        out_dir=sys.argv[3] if len(sys.argv) > 3 else "."
        convert_all(in_dir, out_dir)
        return

    if sys.argv[1] == "--fix-stats":
        if len(sys.argv) < 3:
            print("Usage: python convert_model_rewrite.py --fix-stats <input.h> [output.h]")
            sys.exit(1)
        in_path=sys.argv[2]
        out_path=sys.argv[3] if len(sys.argv) > 3 else "feature_stats.h"

        try:
            convert_feature_stats(in_path, out_path)
            print(f"\n>> Feature stats fixed: {out_path}")

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)
        return

    if sys.argv[1] == "--fix-select":

        if len(sys.argv) < 3:
            print("Usage: python convert_model_rewrite.py --fix-select <input.h> [output.h]")
            sys.exit(1)
        in_path=sys.argv[2]
        out_path=sys.argv[3] if len(sys.argv) > 3 else "feature_select.h"

        try:
            convert_feature_select(in_path, out_path)
            print(f"\n>> Feature select fixed: {out_path}")

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        return

    if sys.argv[1] == "--fix-ensemble":

        if len(sys.argv) < 3:
            print("Usage: python convert_model_rewrite.py --fix-ensemble <input.h> [output.h]")
            sys.exit(1)

        in_path=sys.argv[2]
        out_path=sys.argv[3] if len(sys.argv) > 3 else "ensemble_weights.h"

        try:
            convert_ensemble_weights(in_path, out_path)
            print(f"\n>> Ensemble weights fixed: {out_path}")

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)
        return

    in_path=sys.argv[1]

    if len(sys.argv) > 2:
        out_path=sys.argv[2]

    else:
        stem=Path(in_path).stem
        out_path=f"{stem}_header.h"

    in_dir=Path(in_path).parent
    try_load_feature_select(str(in_dir))

    try:
        if FS["loaded"]:
            scaling="robust" if FS["uses_robust"] else "z-score"
            print(f"Feature selection: {FS['selected_count']} features "
                f"from {FS['original_count']} ({scaling})")

        convert_one_model(in_path, out_path)
        print(f"\nConversion complete: {out_path}")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__=="__main__":
    main()
