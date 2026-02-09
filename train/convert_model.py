"""
Usage: python convert_model.py <model.bin> <output.h>
"""

import struct
import sys
from pathlib import Path
from datetime import datetime
from collections import deque
from abc import ABC, abstractmethod

#cfg
CLASS_NAMES = ["camomile",
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
    "sweet cherry"]

FEATURE_NAMES = ["temp1","temp2","hum1","hum2", "pres1","pres2", "gas1_0", "gas2_0", "delta_temp", "delta_hum", "delta_pres", "delta_gas"]

#magic numbers
MAGIC_RF = 0x52464D4C  #"RFML"
MAGIC_DT = 0x44544D4C  #"DTML"
MAGIC_KNN = 0x4B4E4E4C  #"KNNL"

class BaseModelConverter(ABC):
    """Abstract base class for model converters"""

    def __init__(self):
        self.feature_count = 12
        self.num_classes = len(CLASS_NAMES)
    
    @abstractmethod
    def load_binary(self,path):
        """Load binary model from file"""
        pass

    @abstractmethod
    def generate_header(self,path):
        """Generate C++ header file from model"""
        pass

    @abstractmethod
    def print_summary(self):
        """Print model summary to console"""
        pass

    def _write_file_header(self, f, model, info):
        f.write("// Auto-generated C++ header file\n")
        f.write(f"// Model Type: {model}\n")
        f.write(f"// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if info:
            f.write(f"// Info: {info}\n\n")

    def _write_class_names(self,f,prefix):
        """Write class names array"""
        f.write("// Class Names\n")
        f.write(f"static const char* {prefix}_CLASS_NAMES[{self.num_classes}] = {{\n")
        for i, name in enumerate(CLASS_NAMES):
            comma = "," if i < self.num_classes - 1 else ""
            f.write(f'    "{name}"{comma}\n')
        f.write("};\n\n")

    def _write_progmem_macros(self,f):
        """Progmem compatability macros"""
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
        f.write("  #ifndef pgm_read_float\n")
        f.write("    #define pgm_read_float(addr) (*(const float *)(addr))\n")
        f.write("  #endif\n")
        f.write("  #ifndef memcpy_P\n")
        f.write("    #define memcpy_P(dest, src, n) memcpy((dest), (src), (n))\n")
        f.write("  #endif\n")
        f.write("#endif\n\n")

#===========================================================================================================
#Random Forest Converter
#===========================================================================================================
class RFModelConverter(BaseModelConverter):
    """Random forest model converter"""
    MAGIC = MAGIC_RF
    VERSION = 1

    def __init__(self):
        super().__init__()
        self.num_trees = 0
        self.max_depth = 0
        self.min_samples = 0
        self.feature_subset_ratio = 0.0
        self.feature_subset_size = 0
        self.oob_err = 0.0
        self.feature_importance = []
        self.trees = []
        self.flat_nodes = []
        self.tree_offsets = []

    def load_binary(self, path):
        """load rf bin"""
        with open(path, "rb") as f:
            magic,ver = struct.unpack("<IH", f.read(6))

            if magic != self.MAGIC:
                print(f"Read Magic: {hex(magic)}, Expected: {hex(self.MAGIC)}")
                raise ValueError("Invalid RF model file")
            if ver != self.VERSION:
                raise ValueError("Unsupported RF model version")
            
            #read paramaters
            self.num_trees = struct.unpack("<H", f.read(2))[0]
            self.max_depth = struct.unpack("<B", f.read(1))[0]
            self.min_samples = struct.unpack('<B', f.read(1))[0]
            self.feature_subset_ratio = struct.unpack('<f', f.read(4))[0]
            self.feature_count = struct.unpack('<H', f.read(2))[0]
            self.feature_subset_size = struct.unpack('<H', f.read(2))[0]
            self.oob_error = struct.unpack('<f', f.read(4))[0]

            #feature importance
            self.feature_importance=[]
            for _ in range(self.feature_count):
                importance = struct.unpack("<f",f.read(4))[0]
                self.feature_importance.append(importance)

            #read trees
            self.trees = []
            for _ in range(struct.unpack('<H', f.read(2))[0]):
                tree = self._read_tree(f)
                self.trees.append(tree)

        self._flatten_trees()

        print("Model loaded successfully.")

    def _read_tree(self,f):
        """read node recursively"""
        if not struct.unpack("<B", f.read(1))[0]:
            print("Could not read tree node") 
            return None
        
        #get data
        label = struct.unpack('<B', f.read(1))[0]
        feature_index = struct.unpack('<i', f.read(4))[0]
        threshold = struct.unpack('<f', f.read(4))[0]

        node = {
            "label": label,
            "feature_index": feature_index,
            "threshold": threshold,
            "left": self._read_tree(f),
            "right": self._read_tree(f)
        }
        return node
    
    def _flatten_trees(self):
        self.flat_nodes = []
        self.tree_offsets = []

        for tree in self.trees:
            self.tree_offsets.append(len(self.flat_nodes))
            self._flatten_tree(tree)

    def _flatten_tree(self, tree):
        """flatten using BFS"""
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
            left_idx =-1
            right_idx = -1

            if node["left"]:
                left_idx = offset + node_idx[id(node["left"])]
            if node["right"]:
                right_idx = offset + node_idx[id(node["right"])]

            flat_node = {
                'feature_index': node['feature_index'],
                'label': node['label'],
                'threshold': node['threshold'],
                'left_child': left_idx,
                'right_child': right_idx
            }
            self.flat_nodes.append(flat_node)

    def generate_header(self, output_path):
        """Generate C header file for RF"""
        total_nodes = len(self.flat_nodes)
        estimated_size = total_nodes * 12
        
        with open(output_path, 'w') as f:
            self._write_file_header(f, "Random Forest", 
                f"Trees: {len(self.trees)}, Nodes: {total_nodes}, OOB Error: {self.oob_error * 100:.2f}%")
            
            f.write("#ifndef RF_MODEL_DATA_H\n")
            f.write("#define RF_MODEL_DATA_H\n\n")
            f.write("#include <stdint.h>\n")
            f.write("#include <string.h>\n\n")
            
            self._write_progmem_macros(f)
            
            f.write("// Model Parameters\n")
            f.write(f"#define RF_NUM_TREES {len(self.trees)}\n")
            f.write(f"#define RF_MAX_DEPTH {self.max_depth}\n")
            f.write(f"#define RF_FEATURE_COUNT {self.feature_count}\n")
            f.write(f"#define RF_TOTAL_NODES {total_nodes}\n")
            f.write(f"#define RF_NUM_CLASSES {self.num_classes}\n")
            f.write(f"#define RF_OOB_ERROR {self.oob_error:.6f}f\n\n")
            
            self._write_class_names(f, "RF")
            
            f.write("// Feature Importance (normalized)\n")
            f.write(f"static const float RF_FEATURE_IMPORTANCE[{self.feature_count}] PROGMEM = {{\n    ")
            for i, imp in enumerate(self.feature_importance):
                f.write(f"{imp:.6f}f")
                if i < self.feature_count - 1:
                    f.write(", ")
                if (i + 1) % 6 == 0 and i < self.feature_count - 1:
                    f.write("\n    ")
            f.write("\n};\n\n")
            
            f.write("// Node Structure\n")
            f.write("typedef struct {\n")
            f.write("    int8_t featureIndex;\n")
            f.write("    uint8_t label;\n")
            f.write("    float threshold;\n")
            f.write("    int16_t leftChild;\n")
            f.write("    int16_t rightChild;\n")
            f.write("} rf_node_t;\n\n")
        
            f.write(f"static const rf_node_t RF_NODES[{total_nodes}] PROGMEM = {{\n")
            
            current_tree = 0
            for i, node in enumerate(self.flat_nodes):
                if i in self.tree_offsets:
                    tree_idx = self.tree_offsets.index(i)
                    f.write(f"    // Tree {tree_idx}\n")
                
                f.write(f"    {{{node['feature_index']}, {node['label']}, ")
                f.write(f"{node['threshold']:.6f}f, ")
                f.write(f"{node['left_child']}, {node['right_child']}}}")
                
                if i < total_nodes - 1:
                    f.write(",")
                f.write("\n")
            
            f.write("};\n\n")
            

            f.write(f"static const uint16_t RF_TREE_ROOTS[{len(self.trees)}] PROGMEM = {{")
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
        """Write RF inference functions"""
        f.write("//===========================================================================\n")
        f.write("// RF Inference Functions\n")
        f.write("//===========================================================================\n\n")
        
        #macro
        f.write("#define RF_READ_NODE(idx) ({ \\\n")
        f.write("    rf_node_t _n; \\\n")
        f.write("    memcpy_P(&_n, &RF_NODES[idx], sizeof(rf_node_t)); \\\n")
        f.write("    _n; })\n\n")
        
        #single tree prediction
        f.write("static inline uint8_t rf_predict_tree(uint16_t treeIdx, const float* features) {\n")
        f.write("    uint16_t nodeIdx = pgm_read_word(&RF_TREE_ROOTS[treeIdx]);\n")
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
        
        #forest prediction
        f.write("static inline uint8_t rf_predict(const float* features) {\n")
        f.write(f"    uint16_t votes[{self.num_classes}] = {{0}};\n")
        f.write("    for (uint16_t t = 0; t < RF_NUM_TREES; t++) {\n")
        f.write("        uint8_t pred = rf_predict_tree(t, features);\n")
        f.write(f"        if (pred < {self.num_classes}) votes[pred]++;\n")
        f.write("    }\n")
        f.write("    uint8_t best = 0;\n")
        f.write("    uint16_t maxVotes = 0;\n")
        f.write(f"    for (uint8_t i = 0; i < {self.num_classes}; i++) {{\n")
        f.write("        if (votes[i] > maxVotes) { maxVotes = votes[i]; best = i; }\n")
        f.write("    }\n")
        f.write("    return best;\n")
        f.write("}\n\n")
        
        #confidence
        f.write("static inline uint8_t rf_predict_with_confidence(const float* features, float* confidence) {\n")
        f.write(f"    uint16_t votes[{self.num_classes}] = {{0}};\n")
        f.write("    for (uint16_t t = 0; t < RF_NUM_TREES; t++) {\n")
        f.write("        uint8_t pred = rf_predict_tree(t, features);\n")
        f.write(f"        if (pred < {self.num_classes}) votes[pred]++;\n")
        f.write("    }\n")
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
        f.write("    }\n")
        f.write("\n")
        f.write("    // Laplace smoothing with vote margin and OOB-based calibration to dampen overconfident outputs\n")
        f.write("    const float laplaceTop = (float)(maxVotes + 1) / (float)(RF_NUM_TREES + RF_NUM_CLASSES);\n")
        f.write("    float margin = (float)(maxVotes - secondVotes) / (float)RF_NUM_TREES;\n")
        f.write("    if (margin < 0.0f) margin = 0.0f;\n")
        f.write("    const float calibration = 1.0f - RF_OOB_ERROR;\n")
        f.write("\n")
        f.write("    *confidence = laplaceTop * (0.5f + 0.5f * margin) * calibration;\n")
        f.write("    return best;\n")
        f.write("}\n\n")
        
        #get class name
        f.write("static inline const char* rf_get_class_name(uint8_t idx) {\n")
        f.write(f"    return (idx < {self.num_classes}) ? RF_CLASS_NAMES[idx] : \"unknown\";\n")
        f.write("}\n\n")
    
    def print_summary(self):
        """Print RF model summary"""
        print("\n=== Random Forest Model Summary ===")
        print(f"Trees: {len(self.trees)}")
        print(f"Total Nodes: {len(self.flat_nodes)}")
        print(f"Max Depth: {self.max_depth}")
        print(f"Features: {self.feature_count}")
        print(f"OOB Error: {self.oob_error * 100:.2f}%")
        print("\nFeature Importance:")

        for i, imp in enumerate(self.feature_importance):
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
            bar = "█" * int(imp * 20)
            print(f"  [{i:2d}] {name:12s} {imp:.4f} {bar}")


#===========================================================================================================
#Decision Tree Converter
#===========================================================================================================

class DTModelConverter(BaseModelConverter):
    """Decision Tree model converter"""
    
    MAGIC = MAGIC_DT
    VERSION = 2
    
    def __init__(self):
        super().__init__()
        self.max_depth = 0
        self.min_samples = 0
        self.tree = None
        self.flat_nodes = []
    
    def load_binary(self, filepath):
        """Load binary DT model file"""
        with open(filepath, 'rb') as f:
            magic, version = struct.unpack('<IH', f.read(6))
            
            if magic != self.MAGIC:
                raise ValueError(f"Invalid DT magic number: {hex(magic)}")
            if version not in (1, self.VERSION):
                raise ValueError(f"Unsupported DT version: {version}")
            
            self.feature_count = struct.unpack('<H', f.read(2))[0]
            self.max_depth = struct.unpack('<B', f.read(1))[0]
            self.min_samples = struct.unpack('<B', f.read(1))[0]
            
            self.tree = self._read_tree(f, version)
        
        self._flatten_tree()
        
        print(f"Loaded DT model: {len(self.flat_nodes)} nodes, {self.feature_count} features")
    
    def _read_tree(self, f, version):
        """Recursively read tree nodes"""
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
        
        node = {
            'label': label,
            'feature_index': feature_index,
            'threshold': threshold,
            'majority_count': majority_count,
            'total_samples': total_samples,
            'left': self._read_tree(f, version),
            'right': self._read_tree(f, version)
        }
        return node

    def _flatten_tree(self):
        """Flatten tree to array"""
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
        """Generate C header for DT"""
        total_nodes = len(self.flat_nodes)
        
        with open(output_path, 'w') as f:
            self._write_file_header(f, "Decision Tree", f"Nodes: {total_nodes}")    
            
            f.write("#ifndef DT_MODEL_DATA_H\n")
            f.write("#define DT_MODEL_DATA_H\n\n")
            f.write("#include <stdint.h>\n")
            f.write("#include <string.h>\n\n")  
            
            self._write_progmem_macros(f)
            
            f.write("// Model Parameters\n")
            f.write(f"#define DT_FEATURE_COUNT {self.feature_count}\n")
            f.write(f"#define DT_TOTAL_NODES {total_nodes}\n")
            f.write(f"#define DT_NUM_CLASSES {self.num_classes}\n")
            f.write(f"#define DT_MAX_DEPTH {self.max_depth}\n")
            f.write(f"#define DT_HAS_CONFIDENCE 1\n\n")
            
            self._write_class_names(f, "DT")
            
            #struct
            f.write("typedef struct {\n")
            f.write("    int8_t featureIndex;\n")
            f.write("    uint8_t label;\n")
            f.write("    float threshold;\n")   
            f.write("    uint16_t majorityCount;\n")
            f.write("    uint16_t totalSamples;\n")
            f.write("    int16_t leftChild;\n")
            f.write("    int16_t rightChild;\n")
            f.write("} dt_node_t;\n\n")
            
            #arr
            f.write(f"static const dt_node_t DT_NODES[{total_nodes}] PROGMEM = {{\n")
            for i, node in enumerate(self.flat_nodes):
                f.write(f"    {{{node['feature_index']}, {node['label']}, ")
                f.write(f"{node['threshold']:.6f}f, ")
                f.write(f"{node['majority_count']}, {node['total_samples']}, ")
                f.write(f"{node['left_child']}, {node['right_child']}}}")
                if i < total_nodes - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")
            
            self._write_dt_inference_functions(f)
            
            f.write("#endif // DT_MODEL_DATA_H\n")
        print(f"DT header generated: {output_path}")


    
    def _write_dt_inference_functions(self, f):
        """Write DT inference functions"""
        f.write("//===========================================================================\n")
        f.write("// DT Inference Functions\n")
        f.write("//===========================================================================\n\n")
        
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

        f.write("static inline uint8_t dt_predict_with_confidence(const float* features, float* confidence_out) {\n")
        f.write("    uint16_t nodeIdx = 0;\n")
        f.write("    dt_node_t node = DT_READ_NODE(nodeIdx);\n")
        f.write("    while (node.featureIndex >= 0) {\n")
        f.write("        if (features[node.featureIndex] < node.threshold) {\n")
        f.write("            nodeIdx = node.leftChild;\n")
        f.write("        } else {\n")
        f.write("            nodeIdx = node.rightChild;\n")
        f.write("        }\n")
        f.write("        node = DT_READ_NODE(nodeIdx);\n")
        f.write("    }\n")
        f.write("    if (confidence_out) {\n")
        f.write("        // Laplace-smoothed leaf confidence with stronger prior and heavier support damping\n")
        f.write("        const float total = (float)node.totalSamples;\n")
        f.write("        const float majority = (float)node.majorityCount;\n")
        f.write("        const float laplace = (majority + 0.5f) / (total + (float)DT_NUM_CLASSES);\n")
        f.write("        const float support = total / (total + 20.0f);\n")
        f.write("        *confidence_out = laplace * support;\n")
        f.write("    }\n")
        f.write("    return node.label;\n")
        f.write("}\n\n")
        
        f.write("static inline const char* dt_get_class_name(uint8_t idx) {\n")
        f.write(f"    return (idx < {self.num_classes}) ? DT_CLASS_NAMES[idx] : \"unknown\";\n")
        f.write("}\n\n")
    
    def print_summary(self):
        """Print DT summary"""
        print("\n=== Decision Tree Model Summary ===")
        print(f"Total Nodes: {len(self.flat_nodes)}")
        print(f"Max Depth: {self.max_depth}")
        print(f"Features: {self.feature_count}")


#===========================================================================================================
#kNN converter
#===========================================================================================================
class KNNModelConverter(BaseModelConverter):
    """KNN model converter"""
    
    MAGIC = MAGIC_KNN
    VERSION = 1
    
    def __init__(self):
        super().__init__()
        self.k = 5
        self.sample_count = 0
        self.samples = []

    def load_binary(self, filepath):
        """Load binary KNN model file"""
        with open(filepath, 'rb') as f:
            magic, version = struct.unpack('<IH', f.read(6))
            
            if magic != self.MAGIC:
                raise ValueError(f"Invalid KNN magic number: {hex(magic)}")
            if version != self.VERSION:
                raise ValueError(f"Unsupported KNN version: {version}")

            self.k = struct.unpack('<B', f.read(1))[0]
            self.feature_count = struct.unpack('<H', f.read(2))[0]
            self.sample_count = struct.unpack('<H', f.read(2))[0]
            
            self.samples = []
            for _ in range(self.sample_count):
                label = struct.unpack('<B', f.read(1))[0]
                features = []
                for _ in range(self.feature_count):
                    feat = struct.unpack('<f', f.read(4))[0]
                    features.append(feat)
                self.samples.append({'label': label, 'features': features})
        
        print(f"Loaded KNN model: K={self.k}, {self.sample_count} samples, {self.feature_count} features")
    

    def generate_header(self, output_path):
        """Generate c header for knn"""
        with open(output_path, 'w') as f:
            self._write_file_header(f, "KNN", f"K={self.k}, Samples={self.sample_count}")
            
            f.write("#ifndef KNN_MODEL_DATA_H\n")
            f.write("#define KNN_MODEL_DATA_H\n\n")
            f.write("#include <stdint.h>\n")
            f.write("#include <string.h>\n")
            f.write("#include <math.h>\n\n")
            
            self._write_progmem_macros(f)
            
            f.write("// Model Parameters\n")
            f.write(f"#define KNN_K {self.k}\n")
            f.write(f"#define KNN_FEATURE_COUNT {self.feature_count}\n")
            f.write(f"#define KNN_SAMPLE_COUNT {self.sample_count}\n")
            f.write(f"#define KNN_NUM_CLASSES {self.num_classes}\n\n")
            
            self._write_class_names(f, "KNN")
            
            #struct
            f.write("typedef struct {\n")
            f.write("    uint8_t label;\n")
            f.write(f"    float features[{self.feature_count}];\n")
            f.write("} knn_sample_t;\n\n")
            
            #arr
            f.write(f"static const knn_sample_t KNN_SAMPLES[{self.sample_count}] PROGMEM = {{\n")
            for i, sample in enumerate(self.samples):
                f.write(f"    {{{sample['label']}, {{")
                for j, feat in enumerate(sample['features']):
                    f.write(f"{feat:.6f}f")
                    if j < len(sample['features']) - 1:
                        f.write(", ")
                f.write("}}")
                if i < self.sample_count - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")
            
            self._write_knn_inference_functions(f)
            
            f.write("#endif // KNN_MODEL_DATA_H\n")
        
        print(f"KNN header generated: {output_path}")
        estimated_size = self.sample_count * (1 + self.feature_count * 4)
        print(f"  Estimated flash: {estimated_size:,} bytes")


    def _write_knn_inference_functions(self, f):
        """Write KNN inference functions"""
        f.write("//===========================================================================\n")
        f.write("// KNN Inference Functions\n")
        f.write("//===========================================================================\n\n")
        
        #distance
        f.write("static inline float knn_euclidean_distance(const float* a, const float* b) {\n")
        f.write("    float sum = 0.0f;\n")
        f.write(f"    for (uint16_t i = 0; i < {self.feature_count}; i++) {{\n")
        f.write("        float d = a[i] - b[i];\n")
        f.write("        sum += d * d;\n")
        f.write("    }\n")
        f.write("    return sqrtf(sum);\n")
        f.write("}\n\n")
        
        #pred
        f.write("static inline uint8_t knn_predict(const float* features) {\n")
        f.write("    // Simple implementation - find K nearest neighbors\n")
        f.write(f"    float distances[{self.sample_count}];\n")
        f.write(f"    uint8_t labels[{self.sample_count}];\n")
        f.write("    \n")
        f.write("    // Calculate all distances\n")
        f.write(f"    for (uint16_t i = 0; i < {self.sample_count}; i++) {{\n")
        f.write("        knn_sample_t sample;\n")
        f.write("        memcpy_P(&sample, &KNN_SAMPLES[i], sizeof(knn_sample_t));\n")
        f.write("        distances[i] = knn_euclidean_distance(features, sample.features);\n")
        f.write("        labels[i] = sample.label;\n")
        f.write("    }\n")
        f.write("    \n")
        f.write("    // Find K nearest (simple selection)\n")
        f.write(f"    uint16_t votes[{self.num_classes}] = {{0}};\n")
        f.write(f"    for (uint8_t k = 0; k < {self.k}; k++) {{\n")
        f.write("        float minDist = 1e30f;\n")
        f.write("        uint16_t minIdx = 0;\n")
        f.write(f"        for (uint16_t i = 0; i < {self.sample_count}; i++) {{\n")
        f.write("            if (distances[i] < minDist) {\n")
        f.write("                minDist = distances[i];\n")
        f.write("                minIdx = i;\n")
        f.write("            }\n")
        f.write("        }\n")
        f.write(f"        if (labels[minIdx] < {self.num_classes}) votes[labels[minIdx]]++;\n")
        f.write("        distances[minIdx] = 1e30f; // Mark as used\n")
        f.write("    }\n")
        f.write("    \n")
        f.write("    // Find majority class\n")
        f.write("    uint8_t best = 0;\n")
        f.write("    uint16_t maxVotes = 0;\n")
        f.write(f"    for (uint8_t i = 0; i < {self.num_classes}; i++) {{\n")
        f.write("        if (votes[i] > maxVotes) { maxVotes = votes[i]; best = i; }\n")
        f.write("    }\n")
        f.write("    return best;\n")
        f.write("}\n\n")
        
        #confidence
        f.write("static inline uint8_t knn_predict_with_confidence(const float* features, float* confidence) {\n")
        f.write(f"    float distances[{self.sample_count}];\n")
        f.write(f"    uint8_t labels[{self.sample_count}];\n")
        f.write("    \n")
        f.write(f"    for (uint16_t i = 0; i < {self.sample_count}; i++) {{\n")
        f.write("        knn_sample_t sample;\n")
        f.write("        memcpy_P(&sample, &KNN_SAMPLES[i], sizeof(knn_sample_t));\n")
        f.write("        distances[i] = knn_euclidean_distance(features, sample.features);\n")
        f.write("        labels[i] = sample.label;\n")
        f.write("    }\n")
        f.write("    \n")
        f.write(f"    uint16_t votes[{self.num_classes}] = {{0}};\n")
        f.write(f"    for (uint8_t k = 0; k < {self.k}; k++) {{\n")
        f.write("        float minDist = 1e30f;\n")
        f.write("        uint16_t minIdx = 0;\n")
        f.write(f"        for (uint16_t i = 0; i < {self.sample_count}; i++) {{\n")
        f.write("            if (distances[i] < minDist) { minDist = distances[i]; minIdx = i; }\n")
        f.write("        }\n")
        f.write(f"        if (labels[minIdx] < {self.num_classes}) votes[labels[minIdx]]++;\n")
        f.write("        distances[minIdx] = 1e30f;\n")
        f.write("    }\n")
        f.write("    \n")
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
        f.write("    }\n")
        f.write("\n")
        f.write("    // Laplace smoothing plus vote-margin scaling to temper over-confident outputs on OOD inputs\n")
        f.write("    const float alpha = 0.3f;\n")
        f.write("    const float denom = (float)KNN_K + alpha * (float)KNN_NUM_CLASSES;\n")
        f.write("    const float laplaceTop = (maxVotes + alpha) / (denom > 0.0f ? denom : 1.0f);\n")
        f.write("    float margin = (float)(maxVotes - secondVotes) / (float)KNN_K;\n")
        f.write("    if (margin < 0.0f) margin = 0.0f;\n")
        f.write("\n")
        f.write("    *confidence = laplaceTop * (0.5f + 0.5f * margin);\n")
        f.write("    return best;\n")
        f.write("}\n\n")
        
        f.write("static inline const char* knn_get_class_name(uint8_t idx) {\n")
        f.write(f"    return (idx < {self.num_classes}) ? KNN_CLASS_NAMES[idx] : \"unknown\";\n")
        f.write("}\n\n")


    def print_summary(self):
        """Print KNN summary"""
        print("\n=== KNN Model Summary ===")
        print(f"K: {self.k}")
        print(f"Samples: {self.sample_count}")
        print(f"Features: {self.feature_count}")
        
        #distrubution
        class_counts = [0] * self.num_classes
        for sample in self.samples:
            if sample['label'] < self.num_classes:
                class_counts[sample['label']] += 1
        print("\nClass Distribution:")
        for i, count in enumerate(class_counts):
            if count > 0:
                name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
                print(f"  [{i:2d}] {name}: {count}")



#===========================================================================================================
#main sec
#===========================================================================================================
def detect_model_type(filepath):
    """Auto-detect model type from file"""
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
    
    if magic == MAGIC_RF:
        return 'rf', RFModelConverter()
    elif magic == MAGIC_DT:
        return 'dt', DTModelConverter()
    elif magic == MAGIC_KNN:
        return 'knn', KNNModelConverter()
    else:
        raise ValueError(f"Unknown model type. Magic: {hex(magic)}")
    
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nSupported magic numbers:")
        print(f"  Random Forest: {hex(MAGIC_RF)}")
        print(f"  Decision Tree: {hex(MAGIC_DT)}")
        print(f"  KNN:           {hex(MAGIC_KNN)}")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        stem = Path(input_path).stem
        output_path = f"{stem}_data.h"
    
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