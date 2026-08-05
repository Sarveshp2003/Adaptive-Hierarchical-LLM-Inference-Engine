#!/usr/bin/env python3
"""
Real Model Interactive CLI with Adaptive Scheduler
Uses llama-cpp-python for actual inference + simulated scheduler decisions
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Try to import llama_cpp, install if missing
try:
    from llama_cpp import Llama
except ImportError:
    print("📦 Installing llama-cpp-python...")
    os.system("pip install llama-cpp-python --quiet")
    from llama_cpp import Llama

# Scheduler simulation
class AdaptiveScheduler:
    def __init__(self):
        self.decisions = ["prefetch", "evict", "compress", "keep"]
        self.decision_counter = 0
    
    def get_decision(self):
        decision = self.decisions[self.decision_counter % len(self.decisions)]
        self.decision_counter += 1
        return decision

def format_prompt(user_input: str) -> str:
    """Format using Llama-3.2 chat template."""
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def main():
    model_path = os.getenv("ADAPTIVELLM_MODEL_PATH", 
                           r"E:\AdaptiveLLMRuntime\models\Llama-3.2-3B-Instruct-f16.gguf")
    
    # Verify model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║   LLAMA-3.2-3B REAL INFERENCE + SCHEDULER (Python)    ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    
    print(f"📦 Loading model: {Path(model_path).name}")
    print("⏳ This may take a minute on first load...\n")
    
    try:
        # Initialize the real model
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=10,  # GPU if available
            n_threads=8,
            verbose=False,
            n_ctx=2048
        )
        print("✅ Model loaded successfully!")
        print(f"   Context size: 2048 tokens")
        print(f"   Threads: 8\n")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    scheduler = AdaptiveScheduler()
    
    print("📝 Enter prompts (type 'exit' or 'quit' to exit):\n")
    print("─" * 56)
    
    while True:
        try:
            user_input = input("\n🤖 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\n👋 Goodbye!")
                break
            
            print("\n⏳ Processing...")
            start_time = time.time()
            
            # Format and tokenize
            formatted = format_prompt(user_input)
            input_tokens = llm.tokenize(formatted.encode('utf-8'))
            print(f"  [1/4] Tokenizing... ✓ {len(input_tokens)} tokens")
            
            # Generate with scheduler decisions
            print(f"  [2/4] Generating (with scheduler)... ", end="", flush=True)
            
            # Stream generation
            response_text = ""
            token_count = 0
            decision_interval = 4
            
            for chunk in llm(
                formatted,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
                stream=True,
                echo=False
            ):
                if chunk['choices'][0]['text']:
                    text = chunk['choices'][0]['text']
                    response_text += text
                    print(text, end="", flush=True)
                    
                    # Scheduler decision every N tokens
                    token_count += 1
                    if token_count % decision_interval == 0:
                        decision = scheduler.get_decision()
                        print(f" [scheduler:{decision}]", end="", flush=True)
            
            print("\n", end="")
            
            # Compute metrics
            elapsed = time.time() - start_time
            output_tokens = llm.tokenize(response_text.encode('utf-8'))
            
            print(f"  [3/4] Decoding... ✓")
            print(f"  [4/4] Computing metrics... ✓")
            
            # Display results
            print("\n" + "═" * 56)
            print("📊 INFERENCE RESULTS")
            print("═" * 56)
            print(f"Input:      {user_input}")
            print(f"Tokens:     {len(input_tokens)} prompt + {len(output_tokens)} generated")
            print(f"Time:       {elapsed:.2f}s ({elapsed/len(output_tokens)*1000:.1f}ms/token)")
            print(f"Throughput: {len(output_tokens)/elapsed:.1f} tokens/sec")
            print(f"Response:   {response_text[:200]}")
            if len(response_text) > 200:
                print("            ...")
            print("═" * 56)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
