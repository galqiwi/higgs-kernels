#!/usr/bin/env python3
"""
Plot kernel benchmark results
"""
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "matplotlib>=3.5.0",
#     "numpy>=1.20.0",
# ]
# ///

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re

def run_benchmark():
    """Run the benchmark and parse results"""
    print("Running benchmark...")
    result = subprocess.run(['./build/bench_dequant'], capture_output=True, text=True, cwd='.')
    
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed: {result.stderr}")
    
    # Parse the output
    dimensions = []
    times_orig = []
    times_new = []
    
    lines = result.stdout.split('\n')
    current_dim = None
    current_orig_time = None
    
    for line in lines:
        if "Testing with output dimension:" in line:
            dim_match = re.search(r'dimension: (\d+)', line)
            if dim_match:
                current_dim = int(dim_match.group(1))
        elif "Original kernel:" in line:
            # Next line should have the time
            continue
        elif "Average time:" in line and current_orig_time is None:
            time_match = re.search(r'time: ([\d.]+) ms', line)
            if time_match:
                current_orig_time = float(time_match.group(1))
        elif "New kernel:" in line:
            # Next line should have the time
            continue
        elif "Average time:" in line and current_orig_time is not None:
            time_match = re.search(r'time: ([\d.]+) ms', line)
            if time_match:
                current_new_time = float(time_match.group(1))
                # Now we have both times for this dimension
                dimensions.append(current_dim)
                times_orig.append(current_orig_time)
                times_new.append(current_new_time)
                current_orig_time = None  # Reset for next iteration
    
    return dimensions, times_orig, times_new

def create_plot(dimensions, times_orig, times_new):
    """Create and save the plot"""
    plt.figure(figsize=(12, 8))
    
    # Plot with log scale for better visibility
    plt.loglog(dimensions, times_orig, 'bo-', linewidth=2, markersize=8, label='Original Kernel')
    plt.loglog(dimensions, times_new, 'ro-', linewidth=2, markersize=8, label='New Kernel')
    
    plt.xlabel('Output Dimension (elements)', fontsize=12)
    plt.ylabel('Average Kernel Time (ms)', fontsize=12)
    plt.title('Higgs Dequantization Kernel Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add throughput on secondary y-axis
    ax2 = plt.gca().twinx()
    throughput_orig = [dim / time / 1000 for dim, time in zip(dimensions, times_orig)]  # M elements/sec
    throughput_new = [dim / time / 1000 for dim, time in zip(dimensions, times_new)]  # M elements/sec
    ax2.loglog(dimensions, throughput_orig, 'b--', alpha=0.7, label='Original Throughput')
    ax2.loglog(dimensions, throughput_new, 'r--', alpha=0.7, label='New Throughput')
    ax2.set_ylabel('Throughput (M elements/sec)', fontsize=12, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    plt.tight_layout()
    plt.savefig('kernel_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('kernel_performance_comparison.pdf', bbox_inches='tight')
    print("Comparison plots saved as 'kernel_performance_comparison.png' and 'kernel_performance_comparison.pdf'")
    
    # Also create a linear scale plot for better comparison
    plt.figure(figsize=(12, 6))
    plt.plot(dimensions, times_orig, 'bo-', linewidth=2, markersize=8, label='Original Kernel')
    plt.plot(dimensions, times_new, 'ro-', linewidth=2, markersize=8, label='New Kernel')
    plt.xlabel('Output Dimension (elements)', fontsize=12)
    plt.ylabel('Average Kernel Time (ms)', fontsize=12)
    plt.title('Higgs Dequantization Kernel Performance Comparison (Linear Scale)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis with better labels
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('kernel_performance_comparison_linear.png', dpi=300, bbox_inches='tight')
    print("Linear scale comparison plot saved as 'kernel_performance_comparison_linear.png'")
    
    # Print summary statistics
    print("\nBenchmark Results Summary:")
    print("=" * 80)
    print(f"{'Dimension':<12} {'Orig Time':<10} {'New Time':<10} {'Orig GB/s':<10} {'New GB/s':<10} {'Speedup':<10}")
    print("=" * 80)
    for dim, time_orig, time_new in zip(dimensions, times_orig, times_new):
        throughput_orig = dim / time_orig / 1000  # M elements/sec
        throughput_new = dim / time_new / 1000  # M elements/sec
        bandwidth_orig = throughput_orig * 1e-3  # GB/s
        bandwidth_new = throughput_new * 1e-3  # GB/s
        speedup = time_orig / time_new if time_new > 0 else 0
        print(f"{dim:>8}: {time_orig:>8.3f}ms {time_new:>8.3f}ms {bandwidth_orig:>8.2f} {bandwidth_new:>8.2f} {speedup:>8.2f}x")

if __name__ == "__main__":
    try:
        dimensions, times_orig, times_new = run_benchmark()
        create_plot(dimensions, times_orig, times_new)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)