"""
ML Model Binary-to-C-Header Converter

Usage:
    Single file:  python convert_model.py <model.bin> [output.h]
    Batch mode:   python convert_model.py --all [input_dir] [output_dir]
    Fix stats:    python convert_model.py --fix-stats <feature_stats.h> [output.h]

Supported models:
    - Decision Tree       (magic: 0x44544D4C)
    - K-Nearest Neighbors (magic: 0x4B4E4E4C)
    - Random Forest       (magic: 0x52464D4C)
"""

import struct
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import deque
from abc import ABC, abstractmethod


#===========================================================================================================
#cfg
#===========================================================================================================

CLASS_NAMES = [
    "camomile",
    "thoroughly minted infusion",
    "berry burst",
    "darjeeling blend",
    "decaf nutmeg and vanilla",
    "earl grey",
    "english breakfast tea",
    "fresh orange",
    "garden selection (lemon)",
    "green tea",
    "raspberry",
    "sweet cherry"
]

FEATURE_NAMES = [
    "gas1_resp",
    "gas2_resp",
    "gas_cross",
    "gas_diff",
    "d_temp",
    "d_hum",
    "d_pres",
    "log_gas_cross"
]

DEFAULT_FEATURE_COUNT = 8

#magic numbers
MAGIC_RF  = 0x52464D4C  # "RFML"
MAGIC_DT  = 0x44544D4C  # "DTML"
MAGIC_KNN = 0x4B4E4E4C  # "KNNL"

#header map
MODEL_FILE_MAP = {
    'dt_model.bin':  'dt_model_header.h',
    'knn_model.bin': 'knn_model_header.h',
    'rf_model.bin':  'rf_model_header.h',
}


#===========================================================================================================
#base converter
#===========================================================================================================
class BaseModelConverter(ABC):
    """Abstract base class for model converters"""

    def __init__(self):
        self.feature_count = DEFAULT_FEATURE_COUNT
        self.num_classes = len(CLASS_NAMES)

    @abstractmethod
    def load_binary(self, path):
        pass

    @abstractmethod
    def generate_header(self, path):
        pass

    @abstractmethod
    def print_summary(self):
        pass

    def _get_feature_name(self, idx):
        if idx < len(FEATURE_NAMES):
            return FEATURE_NAMES[idx]
        return f"feature_{idx}"

    def _write_file_header(self, f, model, info):
        f.write("// Auto-generated C++ header file\n")
        f.write(f"// Model Type: {model}\n")
        f.write(f"// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"// Feature count: {self.feature_count}\n")
        f.write(f"// Classes: {self.num_classes}\n")
        if info:
            f.write(f"// Info: {info}\n")
        f.write("\n")

    def _write_class_names(self, f, prefix):
        f.write("// Class Names\n")
        f.write(f"static const char* {prefix}_CLASS_NAMES[{self.num_classes}] = {{\n")
        for i, name in enumerate(CLASS_NAMES):
            comma = "," if i < self.num_classes - 1 else ""
            f.write(f'    "{name}"{comma}\n')
        f.write("};\n\n")

    def _write_progmem_macros(self, f):
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

    def _write_feature_validation(self, f, prefix):
        f.write("// Compile-time feature count validation\n")
        f.write(f"#if defined(TOTAL_ML_FEATURES) && TOTAL_ML_FEATURES != {self.feature_count}\n")
        f.write(f'  #error "{prefix} model was trained with {self.feature_count} features '
                f'but TOTAL_ML_FEATURES is different"\n')
        f.write(f"#endif\n\n")


#===========================================================================================================
#DT converter
#===========================================================================================================
class DTModelConverter(BaseModelConverter):
    MAGIC = MAGIC_DT
    VERSION = 2

    def __init__(self):
        super().__init__()
        self.max_depth = 0
        self.min_samples = 0
        self.tree = None
        self.flat_nodes = []

    def load_binary(self, filepath):
        with open(filepath, 'rb') as f:
            magic, version = struct.unpack('<IH', f.read(6))

            if magic != self.MAGIC:
                raise ValueError(f"Invalid DT magic: {hex(magic)}, expected {hex(self.MAGIC)}")
            if version not in (1, self.VERSION):
                raise ValueError(f"Unsupported DT version: {version}")

            self.feature_count = struct.unpack('<H', f.read(2))[0]
            self.max_depth = struct.unpack('<B', f.read(1))[0]
            self.min_samples = struct.unpack('<B', f.read(1))[0]

            self.tree = self._read_tree(f, version)

        self._flatten_tree()
        print(f"Loaded DT model: {len(self.flat_nodes)} nodes, "f"{self.feature_count} features, max depth {self.max_depth}")

    def _read_tree(self, f, version):
        exists = struct.unpack('<B', f.read(1))[0]
        if not exists:
            return None

        label = struct.unpack('<B', f.read(1))[0]
        feature_index = struct.unpack('<i', f.read(4))[0]
        threshold = struct.unpack('<f', f.read(4))[0]

        if version >= 2:
            majority_count = struct.unpack('<H', f.read(2))[0]
            total_samples = struct.unpack('<H', f.read(2))[0]
        else:
            majority_count = 1
            total_samples = 1

        return {
            'label': label,
            'feature_index': feature_index,
            'threshold': threshold,
            'majority_count': majority_count,
            'total_samples': total_samples,
            'left': self._read_tree(f, version),
            'right': self._read_tree(f, version)
        }

    def _flatten_tree(self):
        if self.tree is None:
            return

        self.flat_nodes = []
        queue = deque([self.tree])
        node_list = []
        node_to_idx = {}

        while queue:
            node = queue.popleft()
            idx = len(node_list)
            node_to_idx[id(node)] = idx
            node_list.append(node)

            if node['left']:
                queue.append(node['left'])
            if node['right']:
                queue.append(node['right'])

        for node in node_list:
            left_idx = node_to_idx[id(node['left'])] if node['left'] else -1
            right_idx = node_to_idx[id(node['right'])] if node['right'] else -1

            self.flat_nodes.append({
                'feature_index': node['feature_index'],
                'label': node['label'],
                'threshold': node['threshold'],
                'majority_count': node['majority_count'],
                'total_samples': node['total_samples'],
                'left_child': left_idx,
                'right_child': right_idx
            })

    def generate_header(self, output_path):
        total_nodes = len(self.flat_nodes)
        estimated_size = total_nodes * 16

        with open(output_path, 'w') as f:
            self._write_file_header(f, "Decision Tree", f"Nodes: {total_nodes}")

            f.write("#ifndef DT_MODEL_DATA_H\n")
            f.write("#define DT_MODEL_DATA_H\n\n")
            f.write("#include <stdint.h>\n")
            f.write("#include <string.h>\n\n")

            self._write_progmem_macros(f)
            self._write_feature_validation(f, "DT")

            f.write("// Model Parameters\n")
            f.write(f"#define DT_FEATURE_COUNT {self.feature_count}\n")
            f.write(f"#define DT_TOTAL_NODES {total_nodes}\n")
            f.write(f"#define DT_NUM_CLASSES {self.num_classes}\n")
            f.write(f"#define DT_MAX_DEPTH {self.max_depth}\n")
            f.write("#define DT_HAS_CONFIDENCE 1\n\n")

            self._write_class_names(f, "DT")

            #node struct
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

            #node array
            f.write(f"static const dt_node_t DT_NODES[{total_nodes}] PROGMEM = {{\n")
            for i, node in enumerate(self.flat_nodes):
                is_leaf = node['feature_index'] < 0
                comment = ""
                if is_leaf:
                    lbl = node['label']
                    name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"class_{lbl}"
                    comment = f"  // LEAF: {name} ({node['majority_count']}/{node['total_samples']})"
                else:
                    fname = self._get_feature_name(node['feature_index'])
                    comment = f"  // {fname} < {node['threshold']:.4f}?"

                f.write(f"    {{{node['feature_index']}, {node['label']}, ")
                f.write(f"{node['threshold']:.6f}f, ")
                f.write(f"{node['majority_count']}, {node['total_samples']}, ")
                f.write(f"{node['left_child']}, {node['right_child']}}}")
                if i < total_nodes - 1:
                    f.write(",")
                f.write(f"{comment}\n")
            f.write("};\n\n")

            self._write_dt_inference_functions(f)

            f.write("#endif // DT_MODEL_DATA_H\n")

        print(f"DT header generated: {output_path}")
        print(f"  Estimated flash: {estimated_size:,} bytes")

    def _write_dt_inference_functions(self, f):
        f.write("// ===========================================================================\n")
        f.write("// DT Inference Functions\n")
        f.write("// ===========================================================================\n\n")

        #macro
        f.write("#define DT_READ_NODE(idx) ({ \\\n")
        f.write("    dt_node_t _n; \\\n")
        f.write("    memcpy_P(&_n, &DT_NODES[idx], sizeof(dt_node_t)); \\\n")
        f.write("    _n; })\n\n")

        f.write("static inline uint8_t dt_predict(const float* features) {\n")
        f.write("    uint16_t nodeIdx = 0;\n")
        f.write("    dt_node_t node = DT_READ_NODE(nodeIdx);\n")
        f.write("    \n")
        f.write("    while (node.featureIndex >= 0) {\n")
        f.write("        if (features[node.featureIndex] < node.threshold) {\n")
        f.write("            nodeIdx = node.leftChild;\n")
        f.write("        } else {\n")
        f.write("            nodeIdx = node.rightChild;\n")
        f.write("        }\n")
        f.write("        node = DT_READ_NODE(nodeIdx);\n")
        f.write("    }\n")
        f.write("    return node.label;\n")
        f.write("}\n\n")

        #conf
        f.write("static inline uint8_t dt_predict_with_confidence(const float* features, float* confidence_out) {\n")
        f.write("    uint16_t nodeIdx = 0;\n")
        f.write("    dt_node_t node = DT_READ_NODE(nodeIdx);\n")
        f.write("    \n")
        f.write("    while (node.featureIndex >= 0) {\n")
        f.write("        if (features[node.featureIndex] < node.threshold) {\n")
        f.write("            nodeIdx = node.leftChild;\n")
        f.write("        } else {\n")
        f.write("            nodeIdx = node.rightChild;\n")
        f.write("        }\n")
        f.write("        node = DT_READ_NODE(nodeIdx);\n")
        f.write("    }\n")
        f.write("    \n")
        f.write("    if (confidence_out) {\n")
        f.write("        // Laplace-smoothed leaf confidence with support damping\n")
        f.write("        const float total = (float)node.totalSamples;\n")
        f.write("        const float majority = (float)node.majorityCount;\n")
        f.write("        const float laplace = (majority + 0.5f) / (total + (float)DT_NUM_CLASSES);\n")
        f.write("        const float support = total / (total + 20.0f);\n")
        f.write("        *confidence_out = laplace * support;\n")
        f.write("    }\n")
        f.write("    return node.label;\n")
        f.write("}\n\n")

        #get class
        f.write("static inline const char* dt_get_class_name(uint8_t idx) {\n")
        f.write(f"    return (idx < {self.num_classes}) ? DT_CLASS_NAMES[idx] : \"unknown\";\n")
        f.write("}\n\n")

    def print_summary(self):
        print("\n=== Decision Tree Model Summary ===")
        print(f"  Total Nodes:  {len(self.flat_nodes)}")
        print(f"  Max Depth:    {self.max_depth}")
        print(f"  Min Samples:  {self.min_samples}")
        print(f"  Features:     {self.feature_count}")

        leaf_count = sum(1 for n in self.flat_nodes if n['feature_index'] < 0)
        internal_count = len(self.flat_nodes) - leaf_count
        print(f"  Internal:     {internal_count}")
        print(f"  Leaves:       {leaf_count}")

        feature_usage = {}
        for n in self.flat_nodes:
            fi = n['feature_index']
            if fi >= 0:
                feature_usage[fi] = feature_usage.get(fi, 0) + 1

        if feature_usage:
            print("\n  Feature Usage (splits):")
            for fi in sorted(feature_usage.keys()):
                name = self._get_feature_name(fi)
                count = feature_usage[fi]
                bar = "█" * count
                print(f"    [{fi:2d}] {name:15s} {count:3d} {bar}")


#===========================================================================================================
#KNNConverter
#===========================================================================================================
class KNNModelConverter(BaseModelConverter):
    MAGIC = MAGIC_KNN
    VERSION = 1

    def __init__(self):
        super().__init__()
        self.k = 5
        self.sample_count = 0
        self.samples = []

    def load_binary(self, filepath):
        with open(filepath, 'rb') as f:
            magic, version = struct.unpack('<IH', f.read(6))

            if magic != self.MAGIC:
                raise ValueError(f"Invalid KNN magic: {hex(magic)}, expected {hex(self.MAGIC)}")
            if version != self.VERSION:
                raise ValueError(f"Unsupported KNN version: {version}")

            self.k = struct.unpack('<B', f.read(1))[0]
            self.feature_count = struct.unpack('<H', f.read(2))[0]
            self.sample_count = struct.unpack('<H', f.read(2))[0]

            self.samples = []
            for _ in range(self.sample_count):
                label = struct.unpack('<B', f.read(1))[0]
                features = list(struct.unpack(f'<{self.feature_count}f',
                                              f.read(4 * self.feature_count)))
                self.samples.append({'label': label, 'features': features})

        print(f"Loaded KNN model: K={self.k}, {self.sample_count} samples, "f"{self.feature_count} features")

    def generate_header(self, output_path):
        estimated_size = self.sample_count * (1 + self.feature_count * 4)
        stack_usage = self.k * 8

        with open(output_path, 'w') as f:
            self._write_file_header(f, "K-Nearest Neighbors",
                                    f"K={self.k}, Samples={self.sample_count}")

            f.write("#ifndef KNN_MODEL_DATA_H\n")
            f.write("#define KNN_MODEL_DATA_H\n\n")
            f.write("#include <stdint.h>\n")
            f.write("#include <string.h>\n")
            f.write("#include <math.h>\n\n")

            self._write_progmem_macros(f)
            self._write_feature_validation(f, "KNN")

            f.write("// Model Parameters\n")
            f.write(f"#define KNN_K {self.k}\n")
            f.write(f"#define KNN_FEATURE_COUNT {self.feature_count}\n")
            f.write(f"#define KNN_SAMPLE_COUNT {self.sample_count}\n")
            f.write(f"#define KNN_NUM_CLASSES {self.num_classes}\n\n")

            self._write_class_names(f, "KNN")

            #struct
            f.write("// Sample Structure\n")
            f.write("typedef struct {\n")
            f.write("    uint8_t label;\n")
            f.write(f"    float features[{self.feature_count}];\n")
            f.write("} knn_sample_t;\n\n")

            #neighbor struct
            f.write("// Neighbor Structure (used during inference)\n")
            f.write("typedef struct {\n")
            f.write("    float distance;\n")
            f.write("    uint8_t label;\n")
            f.write("} knn_neighbor_t;\n\n")

            #sample array
            f.write(f"static const knn_sample_t KNN_SAMPLES[{self.sample_count}] PROGMEM = {{\n")
            for i, sample in enumerate(self.samples):
                lbl = sample['label']
                name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"class_{lbl}"
                f.write(f"    {{{lbl}, {{")
                for j, feat in enumerate(sample['features']):
                    f.write(f"{feat:.6f}f")
                    if j < len(sample['features']) - 1:
                        f.write(", ")
                f.write("}}")
                if i < self.sample_count - 1:
                    f.write(",")
                f.write(f"  // {name}\n")
            f.write("};\n\n")

            self._write_knn_inference_functions(f)

            f.write("#endif // KNN_MODEL_DATA_H\n")

        print(f"KNN header generated: {output_path}")
        print(f"  Estimated flash: {estimated_size:,} bytes")
        print(f"  Inference stack: ~{stack_usage} bytes (K={self.k} neighbors)")

    def _write_knn_inference_functions(self, f):
        f.write("// ===========================================================================\n")
        f.write("// KNN Inference Functions\n")
        f.write("// ===========================================================================\n\n")

        #euclidean dist
        f.write("static inline float knn_euclidean_distance(const float* a, const float* b) {\n")
        f.write("    float sum = 0.0f;\n")
        f.write(f"    for (uint16_t i = 0; i < {self.feature_count}; i++) {{\n")
        f.write("        float d = a[i] - b[i];\n")
        f.write("        sum += d * d;\n")
        f.write("    }\n")
        f.write("    return sqrtf(sum);\n")
        f.write("}\n\n")

        #get k
        f.write("static inline void knn_find_neighbors(const float* features, knn_neighbor_t* neighbors) {\n")
        f.write("    // Initialize with worst-case distances\n")
        f.write("    for (uint8_t k = 0; k < KNN_K; k++) {\n")
        f.write("        neighbors[k].distance = 1e30f;\n")
        f.write("        neighbors[k].label = 0;\n")
        f.write("    }\n\n")
        f.write(f"    for (uint16_t i = 0; i < {self.sample_count}; i++) {{\n")
        f.write("        knn_sample_t sample;\n")
        f.write("        memcpy_P(&sample, &KNN_SAMPLES[i], sizeof(knn_sample_t));\n")
        f.write("        float dist = knn_euclidean_distance(features, sample.features);\n\n")
        f.write("        // Insert if closer than current worst neighbor\n")
        f.write("        if (dist < neighbors[KNN_K - 1].distance) {\n")
        f.write("            neighbors[KNN_K - 1].distance = dist;\n")
        f.write("            neighbors[KNN_K - 1].label = sample.label;\n\n")
        f.write("            // Bubble up to maintain sorted order\n")
        f.write("            for (int8_t j = KNN_K - 2; j >= 0; j--) {\n")
        f.write("                if (neighbors[j + 1].distance < neighbors[j].distance) {\n")
        f.write("                    knn_neighbor_t tmp = neighbors[j];\n")
        f.write("                    neighbors[j] = neighbors[j + 1];\n")
        f.write("                    neighbors[j + 1] = tmp;\n")
        f.write("                } else {\n")
        f.write("                    break;\n")
        f.write("                }\n")
        f.write("            }\n")
        f.write("        }\n")
        f.write("    }\n")
        f.write("}\n\n")

        #pred
        f.write("static inline uint8_t knn_predict(const float* features) {\n")
        f.write("    knn_neighbor_t neighbors[KNN_K];\n")
        f.write("    knn_find_neighbors(features, neighbors);\n\n")
        f.write(f"    uint16_t votes[{self.num_classes}] = {{0}};\n")
        f.write("    for (uint8_t k = 0; k < KNN_K; k++) {\n")
        f.write(f"        if (neighbors[k].label < {self.num_classes}) {{\n")
        f.write("            votes[neighbors[k].label]++;\n")
        f.write("        }\n")
        f.write("    }\n\n")
        f.write("    uint8_t best = 0;\n")
        f.write("    uint16_t maxVotes = 0;\n")
        f.write(f"    for (uint8_t i = 0; i < {self.num_classes}; i++) {{\n")
        f.write("        if (votes[i] > maxVotes) {\n")
        f.write("            maxVotes = votes[i];\n")
        f.write("            best = i;\n")
        f.write("        }\n")
        f.write("    }\n")
        f.write("    return best;\n")
        f.write("}\n\n")

        #conf
        f.write("static inline uint8_t knn_predict_with_confidence(const float* features, float* confidence) {\n")
        f.write("    knn_neighbor_t neighbors[KNN_K];\n")
        f.write("    knn_find_neighbors(features, neighbors);\n\n")
        f.write("    // Distance-weighted voting\n")
        f.write(f"    float scores[{self.num_classes}];\n")
        f.write(f"    memset(scores, 0, sizeof(scores));\n")
        f.write("    float totalWeight = 0.0f;\n\n")
        f.write("    for (uint8_t k = 0; k < KNN_K; k++) {\n")
        f.write("        float w = 1.0f / (neighbors[k].distance + 1e-6f);\n")
        f.write(f"        if (neighbors[k].label < {self.num_classes}) {{\n")
        f.write("            scores[neighbors[k].label] += w;\n")
        f.write("            totalWeight += w;\n")
        f.write("        }\n")
        f.write("    }\n\n")
        f.write("    // Find best and second-best\n")
        f.write("    uint8_t best = 0;\n")
        f.write("    float maxScore = 0.0f;\n")
        f.write("    float secondScore = 0.0f;\n")
        f.write(f"    for (uint8_t i = 0; i < {self.num_classes}; i++) {{\n")
        f.write("        if (scores[i] > maxScore) {\n")
        f.write("            secondScore = maxScore;\n")
        f.write("            maxScore = scores[i];\n")
        f.write("            best = i;\n")
        f.write("        } else if (scores[i] > secondScore) {\n")
        f.write("            secondScore = scores[i];\n")
        f.write("        }\n")
        f.write("    }\n\n")
        f.write("    // Confidence: normalized weighted score with margin damping\n")
        f.write("    float rawConf = (totalWeight > 0.0f) ? (maxScore / totalWeight) : 0.0f;\n")
        f.write("    float margin = (totalWeight > 0.0f) ? ((maxScore - secondScore) / totalWeight) : 0.0f;\n")
        f.write("    *confidence = rawConf * (0.5f + 0.5f * margin);\n\n")
        f.write("    return best;\n")
        f.write("}\n\n")

        #get class
        f.write("static inline const char* knn_get_class_name(uint8_t idx) {\n")
        f.write(f"    return (idx < {self.num_classes}) ? KNN_CLASS_NAMES[idx] : \"unknown\";\n")
        f.write("}\n\n")

    def print_summary(self):
        print("\n=== KNN Model Summary ===")
        print(f"  K:        {self.k}")
        print(f"  Samples:  {self.sample_count}")
        print(f"  Features: {self.feature_count}")

        #class dist
        print("\n  Class Distribution:")
        class_counts = [0] * self.num_classes
        for sample in self.samples:
            if sample['label'] < self.num_classes:
                class_counts[sample['label']] += 1

        for i, count in enumerate(class_counts):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
            bar = "█" * count
            print(f"    [{i:2d}] {name:30s} {count:3d} {bar}")

        #memory est
        flash_bytes = self.sample_count * (1 + self.feature_count * 4)
        stack_bytes = self.k * 8
        print(f"\n  Flash estimate:       {flash_bytes:,} bytes")
        print(f"  Inference stack est:  ~{stack_bytes} bytes")
        if flash_bytes > 65536:
            print(f"WARNING: Large model, may be slow on embedded targets")


#===========================================================================================================
#RF converter
#===========================================================================================================
class RFModelConverter(BaseModelConverter):
    MAGIC = MAGIC_RF
    VERSION = 1

    def __init__(self):
        super().__init__()
        self.num_trees = 0
        self.max_depth = 0
        self.min_samples = 0
        self.feature_subset_ratio = 0.0
        self.feature_subset_size = 0
        self.oob_error = 0.0
        self.feature_importance = []
        self.trees = []
        self.flat_nodes = []
        self.tree_offsets = []

    def load_binary(self, path):
        with open(path, "rb") as f:
            magic, ver = struct.unpack("<IH", f.read(6))

            if magic != self.MAGIC:
                raise ValueError(f"Invalid RF magic: {hex(magic)}, expected {hex(self.MAGIC)}")
            if ver != self.VERSION:
                raise ValueError(f"Unsupported RF version: {ver}")

            self.num_trees = struct.unpack("<H", f.read(2))[0]
            self.max_depth = struct.unpack("<B", f.read(1))[0]
            self.min_samples = struct.unpack('<B', f.read(1))[0]
            self.feature_subset_ratio = struct.unpack('<f', f.read(4))[0]
            self.feature_count = struct.unpack('<H', f.read(2))[0]
            self.feature_subset_size = struct.unpack('<H', f.read(2))[0]
            self.oob_error = struct.unpack('<f', f.read(4))[0]

            #feature importance
            self.feature_importance = []
            for _ in range(self.feature_count):
                importance = struct.unpack("<f", f.read(4))[0]
                self.feature_importance.append(importance)

            #trees
            self.trees = []
            tree_count = struct.unpack('<H', f.read(2))[0]
            for _ in range(tree_count):
                tree = self._read_tree(f)
                self.trees.append(tree)

        self._flatten_trees()
        print(f"Loaded RF model: {len(self.trees)} trees, {len(self.flat_nodes)} total nodes, "
              f"{self.feature_count} features, OOB error {self.oob_error * 100:.2f}%")

    def _read_tree(self, f):
        if not struct.unpack("<B", f.read(1))[0]:
            return None

        label = struct.unpack('<B', f.read(1))[0]
        feature_index = struct.unpack('<i', f.read(4))[0]
        threshold = struct.unpack('<f', f.read(4))[0]

        return {
            "label": label,
            "feature_index": feature_index,
            "threshold": threshold,
            "left": self._read_tree(f),
            "right": self._read_tree(f)
        }

    def _flatten_trees(self):
        self.flat_nodes = []
        self.tree_offsets = []

        for tree in self.trees:
            self.tree_offsets.append(len(self.flat_nodes))
            self._flatten_tree(tree)

    def _flatten_tree(self, tree):
        if tree is None:
            return

        queue = deque([tree])
        node_list = []
        node_idx = {}

        while queue:
            node = queue.popleft()
            i = len(node_list)
            node_idx[id(node)] = i
            node_list.append(node)

            if node["left"]:
                queue.append(node["left"])
            if node["right"]:
                queue.append(node["right"])

        offset = len(self.flat_nodes)
        for node in node_list:
            left_idx = -1
            right_idx = -1

            if node["left"]:
                left_idx = offset + node_idx[id(node["left"])]
            if node["right"]:
                right_idx = offset + node_idx[id(node["right"])]

            self.flat_nodes.append({
                'feature_index': node['feature_index'],
                'label': node['label'],
                'threshold': node['threshold'],
                'left_child': left_idx,
                'right_child': right_idx
            })

    def generate_header(self, output_path):
        total_nodes = len(self.flat_nodes)
        estimated_size = total_nodes * 16

        with open(output_path, 'w') as f:
            self._write_file_header(f, "Random Forest",
                                    f"Trees: {len(self.trees)}, Nodes: {total_nodes}, "
                                    f"OOB Error: {self.oob_error * 100:.2f}%")

            f.write("#ifndef RF_MODEL_DATA_H\n")
            f.write("#define RF_MODEL_DATA_H\n\n")
            f.write("#include <stdint.h>\n")
            f.write("#include <string.h>\n\n")

            self._write_progmem_macros(f)
            self._write_feature_validation(f, "RF")

            f.write("// Model Parameters\n")
            f.write(f"#define RF_NUM_TREES {len(self.trees)}\n")
            f.write(f"#define RF_MAX_DEPTH {self.max_depth}\n")
            f.write(f"#define RF_FEATURE_COUNT {self.feature_count}\n")
            f.write(f"#define RF_TOTAL_NODES {total_nodes}\n")
            f.write(f"#define RF_NUM_CLASSES {self.num_classes}\n")
            f.write(f"#define RF_OOB_ERROR {self.oob_error:.6f}f\n\n")

            self._write_class_names(f, "RF")

            #feature importance
            f.write("// Feature Importance (normalized)\n")
            f.write(f"static const float RF_FEATURE_IMPORTANCE[{self.feature_count}] PROGMEM = {{\n")
            for i, imp in enumerate(self.feature_importance):
                name = self._get_feature_name(i)
                f.write(f"    {imp:.6f}f")
                if i < self.feature_count - 1:
                    f.write(",")
                f.write(f"  // [{i}] {name}\n")
            f.write("};\n\n")

            #node
            f.write("// Node Structure\n")
            f.write("typedef struct {\n")
            f.write("    int8_t featureIndex;\n")
            f.write("    uint8_t label;\n")
            f.write("    float threshold;\n")
            f.write("    int32_t leftChild;\n")
            f.write("    int32_t rightChild;\n")
            f.write("} rf_node_t;\n\n")
            f.write(f"static const rf_node_t RF_NODES[{total_nodes}] PROGMEM = {{\n")

            for i, node in enumerate(self.flat_nodes):
                #tree
                if i in self.tree_offsets:
                    tree_idx = self.tree_offsets.index(i)
                    f.write(f"    // --- Tree {tree_idx} ---\n")

                f.write(f"    {{{node['feature_index']}, {node['label']}, ")
                f.write(f"{node['threshold']:.6f}f, ")
                f.write(f"{node['left_child']}, {node['right_child']}}}")

                if i < total_nodes - 1:
                    f.write(",")
                f.write("\n")

            f.write("};\n\n")

            #root offsets
            f.write(f"static const uint32_t RF_TREE_ROOTS[{len(self.trees)}] PROGMEM = {{")
            for i, offset in enumerate(self.tree_offsets):
                if i > 0:
                    f.write(", ")
                if i % 10 == 0:
                    f.write("\n    ")
                f.write(str(offset))
            f.write("\n};\n\n")

            self._write_rf_inference_functions(f)

            f.write("#endif // RF_MODEL_DATA_H\n")

        print(f"RF header generated: {output_path}")
        print(f"  Estimated flash: {estimated_size:,} bytes")

    def _write_rf_inference_functions(self, f):
        f.write("// ===========================================================================\n")
        f.write("// RF Inference Functions\n")
        f.write("// ===========================================================================\n\n")

        #macro
        f.write("#define RF_READ_NODE(idx) ({ \\\n")
        f.write("    rf_node_t _n; \\\n")
        f.write("    memcpy_P(&_n, &RF_NODES[idx], sizeof(rf_node_t)); \\\n")
        f.write("    _n; })\n\n")

        #single tree pred
        f.write("static inline uint8_t rf_predict_tree(uint32_t treeIdx, const float* features) {\n")
        f.write("    uint32_t nodeIdx = pgm_read_dword(&RF_TREE_ROOTS[treeIdx]);\n")
        f.write("    rf_node_t node = RF_READ_NODE(nodeIdx);\n")
        f.write("    \n")
        f.write("    while (node.featureIndex >= 0) {\n")
        f.write("        if (features[node.featureIndex] < node.threshold) {\n")
        f.write("            nodeIdx = node.leftChild;\n")
        f.write("        } else {\n")
        f.write("            nodeIdx = node.rightChild;\n")
        f.write("        }\n")
        f.write("        node = RF_READ_NODE(nodeIdx);\n")
        f.write("    }\n")
        f.write("    return node.label;\n")
        f.write("}\n\n")

        #forest pred
        f.write("static inline uint8_t rf_predict(const float* features) {\n")
        f.write(f"    uint16_t votes[{self.num_classes}] = {{0}};\n")
        f.write("    for (uint32_t t = 0; t < RF_NUM_TREES; t++) {\n")
        f.write("        uint8_t pred = rf_predict_tree(t, features);\n")
        f.write(f"        if (pred < {self.num_classes}) votes[pred]++;\n")
        f.write("    }\n\n")
        f.write("    uint8_t best = 0;\n")
        f.write("    uint16_t maxVotes = 0;\n")
        f.write(f"    for (uint8_t i = 0; i < {self.num_classes}; i++) {{\n")
        f.write("        if (votes[i] > maxVotes) {\n")
        f.write("            maxVotes = votes[i];\n")
        f.write("            best = i;\n")
        f.write("        }\n")
        f.write("    }\n")
        f.write("    return best;\n")
        f.write("}\n\n")

        #conf
        f.write("static inline uint8_t rf_predict_with_confidence(const float* features, float* confidence) {\n")
        f.write(f"    uint16_t votes[{self.num_classes}] = {{0}};\n")
        f.write("    for (uint32_t t = 0; t < RF_NUM_TREES; t++) {\n")
        f.write("        uint8_t pred = rf_predict_tree(t, features);\n")
        f.write(f"        if (pred < {self.num_classes}) votes[pred]++;\n")
        f.write("    }\n\n")
        f.write("    uint8_t best = 0;\n")
        f.write("    uint16_t maxVotes = 0;\n")
        f.write("    uint16_t secondVotes = 0;\n")
        f.write(f"    for (uint8_t i = 0; i < {self.num_classes}; i++) {{\n")
        f.write("        if (votes[i] > maxVotes) {\n")
        f.write("            secondVotes = maxVotes;\n")
        f.write("            maxVotes = votes[i];\n")
        f.write("            best = i;\n")
        f.write("        } else if (votes[i] > secondVotes) {\n")
        f.write("            secondVotes = votes[i];\n")
        f.write("        }\n")
        f.write("    }\n\n")
        f.write("    // Laplace smoothing with vote margin and OOB calibration\n")
        f.write("    const float laplaceTop = (float)(maxVotes + 1) / (float)(RF_NUM_TREES + RF_NUM_CLASSES);\n")
        f.write("    float margin = (float)(maxVotes - secondVotes) / (float)RF_NUM_TREES;\n")
        f.write("    if (margin < 0.0f) margin = 0.0f;\n")
        f.write("    const float calibration = 1.0f - RF_OOB_ERROR;\n\n")
        f.write("    *confidence = laplaceTop * (0.5f + 0.5f * margin) * calibration;\n")
        f.write("    return best;\n")
        f.write("}\n\n")

        #get class
        f.write("static inline const char* rf_get_class_name(uint8_t idx) {\n")
        f.write(f"    return (idx < {self.num_classes}) ? RF_CLASS_NAMES[idx] : \"unknown\";\n")
        f.write("}\n\n")

    def print_summary(self):
        print("\n=== Random Forest Model Summary ===")
        print(f"  Trees:       {len(self.trees)}")
        print(f"  Total Nodes: {len(self.flat_nodes)}")
        print(f"  Max Depth:   {self.max_depth}")
        print(f"  Min Samples: {self.min_samples}")
        print(f"  Features:    {self.feature_count}")
        print(f"  Subset Size: {self.feature_subset_size}")
        print(f"  Subset Ratio:{self.feature_subset_ratio:.2f}")
        print(f"  OOB Error:   {self.oob_error * 100:.2f}%")

        node_counts = []
        for i in range(len(self.tree_offsets)):
            start = self.tree_offsets[i]
            end = self.tree_offsets[i + 1] if i + 1 < len(self.tree_offsets) else len(self.flat_nodes)
            node_counts.append(end - start)

        if node_counts:
            avg_nodes = sum(node_counts) / len(node_counts)
            print(f"  Avg Nodes/Tree: {avg_nodes:.1f}")
            print(f"  Min Nodes/Tree: {min(node_counts)}")
            print(f"  Max Nodes/Tree: {max(node_counts)}")

        if self.feature_importance:
            print("\n  Feature Importance:")
            sorted_features = sorted(enumerate(self.feature_importance),key=lambda x: x[1], reverse=True)
            max_imp = max(self.feature_importance) if self.feature_importance else 1.0
            for idx, imp in sorted_features:
                name = self._get_feature_name(idx)
                bar_len = int((imp / max_imp) * 30) if max_imp > 0 else 0
                bar = "█" * bar_len
                print(f"    [{idx:2d}] {name:15s} {imp:.6f} {bar}")

        #Flash estimate
                estimated_flash = len(self.flat_nodes) * 16 
        print(f"\n  Flash estimate: {estimated_flash:,} bytes")


#===========================================================================================================
#Feature Stats Converter
#===========================================================================================================

def convert_feature_stats(input_path, output_path):
    """
    Read trainer-generated feature_stats.h (which may use FEATURE_MEANS/FEATURE_STDS)
    and regenerate with correct FEATURE_MEAN/FEATURE_STD naming to match ml_inference.cpp.
    """
    with open(input_path, 'r') as f:
        content = f.read()

    # Parse either FEATURE_MEANS or FEATURE_MEAN
    means_match = re.search(r'FEATURE_MEAN[S]?$$.*?$$\s*=\s*\{([^}]+)\}', content)
    stds_match = re.search(r'FEATURE_STD[S]?$$.*?$$\s*=\s*\{([^}]+)\}', content)

    if not means_match or not stds_match:
        raise ValueError("Could not parse feature stats from input file")

    means = [float(x.strip()) for x in means_match.group(1).split(',')]
    stds = [float(x.strip()) for x in stds_match.group(1).split(',')]

    count = len(means)
    if len(stds) != count:
        raise ValueError(f"Means count ({len(means)}) != stds count ({len(stds)})")

    if count != DEFAULT_FEATURE_COUNT:
        print(f"WARNING: Feature stats has {count} features, "f"expected {DEFAULT_FEATURE_COUNT}")

    with open(output_path, 'w') as f:
        f.write("// Auto-generated feature statistics for z-score normalization\n")
        f.write(f"// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"// Features: {count}\n")
        f.write("// Names match ml_inference.cpp: FEATURE_MEAN[] and FEATURE_STD[]\n\n")

        f.write("#ifndef FEATURE_STATS_H\n")
        f.write("#define FEATURE_STATS_H\n\n")

        f.write(f"#define FEATURE_STATS_COUNT {count}\n\n")

        #compile-time validation
        f.write(f"#if defined(TOTAL_ML_FEATURES) && TOTAL_ML_FEATURES != {count}\n")
        f.write(f'  #error "Feature stats has {count} features but TOTAL_ML_FEATURES differs"\n')
        f.write(f"#endif\n\n")

        f.write(f"static const float FEATURE_MEAN[{count}] = {{\n")
        for i, m in enumerate(means):
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
            comma = "," if i < count - 1 else ""
            f.write(f"    {m:.8f}f{comma}  // [{i}] {name}\n")
        f.write("};\n\n")

        f.write(f"static const float FEATURE_STD[{count}] = {{\n")
        for i, s in enumerate(stds):
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
            comma = "," if i < count - 1 else ""
            f.write(f"    {s:.8f}f{comma}  // [{i}] {name}\n")
        f.write("};\n\n")

        f.write("#endif // FEATURE_STATS_H\n")

    print(f"Feature stats header generated: {output_path}")
    print(f"  Features: {count}")
    for i in range(count):
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
        print(f"    [{i}] {name:15s}  mean={means[i]:12.6f}  std={stds[i]:12.6f}")


#===========================================================================================================
#detection and batch conversion
#===========================================================================================================

def detect_model_type(filepath):
    """Auto-detect model type from file magic number"""
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]

    if magic == MAGIC_DT:
        return 'dt', DTModelConverter()
    elif magic == MAGIC_KNN:
        return 'knn', KNNModelConverter()
    elif magic == MAGIC_RF:
        return 'rf', RFModelConverter()
    else:
        raise ValueError(f"Unknown model type. Magic: {hex(magic)}")


def convert_all(input_dir=".", output_dir="."):
    """Batch convert all model binary files and feature stats found in a directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    failed = 0

    #convert binaries
    for bin_file, header_file in MODEL_FILE_MAP.items():
        input_path = input_dir / bin_file
        output_path = output_dir / header_file

        if not input_path.exists():
            print(f"Skipping {bin_file} (not found)")
            continue

        try:
            model_type, converter = detect_model_type(str(input_path))
            print(f"\n{'=' * 60}")
            print(f"Converting: {bin_file} → {header_file}")
            print(f"{'=' * 60}")

            converter.load_binary(str(input_path))
            converter.print_summary()
            converter.generate_header(str(output_path))
            converted += 1

        except Exception as e:
            print(f"  ✗ Failed to convert {bin_file}: {e}")
            failed += 1

    #convert feature_stats.h if present
    stats_input = input_dir / "feature_stats.h"
    stats_output = output_dir / "feature_stats.h"

    if stats_input.exists():
        try:
            print(f"\n{'=' * 60}")
            print(f"Converting: feature_stats.h (fixing naming)")
            print(f"{'=' * 60}")
            convert_feature_stats(str(stats_input), str(stats_output))
            converted += 1
        except Exception as e:
            print(f"  ✗ Failed to convert feature_stats.h: {e}")
            failed += 1
    else:
        print(f"\nSkipping feature_stats.h (not found in {input_dir})")

    # Summary
    total = converted + failed
    print(f"\n{'=' * 60}")
    print(f"Batch conversion complete")
    print(f"  Converted: {converted}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {len(MODEL_FILE_MAP) + 1 - total}")
    print(f"{'=' * 60}")

    return converted


#===========================================================================================================
#   Main
#===========================================================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nSupported magic numbers:")
        print(f"  Decision Tree:  {hex(MAGIC_DT)}")
        print(f"  KNN:            {hex(MAGIC_KNN)}")
        print(f"  Random Forest:  {hex(MAGIC_RF)}")
        print()
        print("Examples:")
        print("  python convert_model.py rf_model.bin rf_model_header.h")
        print("  python convert_model.py --all ./build ./src/model_headers")
        print("  python convert_model.py --fix-stats feature_stats.h feature_stats_fixed.h")
        sys.exit(1)

    #batch mode
    if sys.argv[1] == '--all':
        input_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
        convert_all(input_dir, output_dir)
        return

    #fix stats mode
    if sys.argv[1] == '--fix-stats':
        if len(sys.argv) < 3:
            print("Usage: python convert_model.py --fix-stats <input.h> [output.h]")
            sys.exit(1)
        input_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "feature_stats.h"
        try:
            convert_feature_stats(input_path, output_path)
            print(f"\n✓ Feature stats fixed: {output_path}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    input_path = sys.argv[1]

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        stem = Path(input_path).stem
        output_path = f"{stem}_header.h"

    try:
        model_type, converter = detect_model_type(input_path)
        print(f"Detected model type: {model_type.upper()}")

        converter.load_binary(input_path)
        converter.print_summary()
        converter.generate_header(output_path)

        print(f"\n✓ Conversion complete: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()