#!/usr/bin/env python3
"""
Extract GGUF tensors and save as adaptive archive.
Uses the gguf module from llama.cpp (built-in to repo).
"""

import sys
import struct
import numpy as np
from pathlib import Path

# Add llama.cpp gguf-py to path
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "llama.cpp" / "gguf-py"))

try:
    from gguf import GGUFReader
except ImportError as e:
    print(f"ERROR: Could not import gguf.GGUFReader: {e}")
    print("Make sure llama.cpp/gguf-py is in the path")
    sys.exit(1)

def extract_and_convert(gguf_path, output_prefix):
    """Extract GGUF tensors and save as adaptive format."""
    print(f"[*] Loading GGUF file: {gguf_path}")
    
    try:
        gguf_data = GGUFReader(gguf_path)
    except Exception as e:
        print(f"[!] Failed to read GGUF: {e}")
        raise
    
    print(f"[*] Found {len(gguf_data.tensors)} tensors")
    
    # Open output files
    adaptive_path = Path(output_prefix + ".adaptive")
    weights_path = Path(output_prefix + ".weights")
    index_path = Path(output_prefix + ".weights.idx")
    
    print(f"[*] Writing to {adaptive_path}")
    
    with open(adaptive_path, 'w') as adap, \
         open(weights_path, 'wb') as wbin, \
         open(index_path, 'w') as widx:
        
        # Write header
        adap.write(f"{Path(gguf_path).stem}\n")  # name
        adap.write("gguf\n")  # format
        adap.write("3\n")  # version
        adap.write(f"{len(gguf_data.tensors)}\n")  # num layers
        
        offset = 0
        for i, tensor in enumerate(gguf_data.tensors):
            tensor_name = tensor.name
            
            if i % 50 == 0:
                print(f"[*] Processing tensor {i+1}/{len(gguf_data.tensors)}: {tensor_name}")
            
            try:
                # Get tensor data
                tensor_data = tensor.data
                if tensor_data is None:
                    print(f"    [!] Skipping {tensor_name}: no data")
                    continue
                
                # Get shape and dtype
                shape = list(tensor.shape)
                dtype = tensor.tensor_type
                
                # Convert to numpy array if needed
                if isinstance(tensor_data, np.ndarray):
                    arr = tensor_data
                else:
                    arr = np.asarray(tensor_data)
                
                # Flatten and ensure float32
                flat_arr = arr.flatten().astype(np.float32)
                float_count = len(flat_arr)
                
                # Extract layer ID from tensor name
                layer_id = 0
                if "layers" in tensor_name or "layer" in tensor_name:
                    parts = tensor_name.split(".")
                    for part in parts:
                        if part.isdigit():
                            layer_id = int(part)
                            break
                
                # Write to adaptive file
                adap.write(f"{layer_id}\n")  # layer_id
                adap.write(f"{tensor_name}\n")  # name
                adap.write(",".join(str(s) for s in shape) + "\n")  # shape
                
                # Write weights to binary file
                weight_bytes = flat_arr.tobytes()
                wbin.write(weight_bytes)
                
                # Write index entry
                widx.write(f"{layer_id} {offset} {float_count}\n")
                
                offset += len(weight_bytes)
                
                # Empty weights line in adaptive (external weights)
                adap.write("\n")
            except Exception as e:
                print(f"    [!] Error processing {tensor_name}: {e}")
                continue
    
    print(f"[+] Done! Generated:")
    print(f"    {adaptive_path}")
    print(f"    {weights_path}")
    print(f"    {index_path}")
    print(f"[+] Total offset: {offset} bytes")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: extract_gguf.py <gguf_file> [output_prefix]")
        sys.exit(1)
    
    gguf_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else str(Path(gguf_file).parent / Path(gguf_file).stem)
    
    if not Path(gguf_file).exists():
        print(f"ERROR: File not found: {gguf_file}")
        sys.exit(1)
    
    try:
        extract_and_convert(gguf_file, output_prefix)
        print("\n[OK] Conversion succeeded")
    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

