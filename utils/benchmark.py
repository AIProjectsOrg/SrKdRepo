"""
Benchmarking utilities for model performance and speed testing.
"""
import torch
import time
try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
except ImportError:
    print("Warning: fvcore not found. FLOPs analysis will not be available.")
    FlopCountAnalysis = None
    parameter_count_table = None


class ModelBenchmark:
    """Handles model benchmarking for speed and efficiency"""
    
    def __init__(self, device):
        self.device = device
    
    def benchmark_inference_time(self, model, input_tensor, model_name="Model", warmup=10, runs=50):
        """
        Benchmark inference time and throughput
        
        Args:
            model: Model to benchmark
            input_tensor: Input tensor for benchmarking
            model_name: Name for logging
            warmup: Number of warmup iterations
            runs: Number of timed runs
        """
        model.eval().to(self.device)
        input_tensor = input_tensor.to(self.device)

        # Warm-up
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()

        avg_time = (end_time - start_time) / runs
        throughput = 1.0 / avg_time
        
        print(f"\n==== {model_name} Benchmark ====")
        print(f"Inference Time: {avg_time * 1000:.2f} ms/image")
        print(f"Throughput: {throughput:.2f} images/sec")
        
        return avg_time, throughput
    
    def measure_flops_and_params(self, model, input_tensor, model_name="Model"):
        """
        Estimate FLOPs and parameters using fvcore
        
        Args:
            model: Model to analyze
            input_tensor: Input tensor for analysis
            model_name: Name for logging
        """
        if FlopCountAnalysis is None or parameter_count_table is None:
            print("fvcore not available. Skipping FLOPs analysis.")
            return None
            
        model.eval()
        flops = FlopCountAnalysis(model, input_tensor)
        
        print(f"\n==== {model_name} FLOPs & Params ====")
        print(parameter_count_table(model))
        print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
        
        return flops.total()
    
    def full_benchmark(self, model, input_tensor, model_name="Model", warmup=10, runs=50):
        """
        Run complete benchmark including time and efficiency metrics
        
        Args:
            model: Model to benchmark
            input_tensor: Input tensor
            model_name: Name for logging
            warmup: Warmup iterations
            runs: Timed runs
        """
        print(f"\n{'='*50}")
        print(f"Benchmarking {model_name}")
        print(f"{'='*50}")
        
        # Time benchmark
        avg_time, throughput = self.benchmark_inference_time(
            model, input_tensor, model_name, warmup, runs
        )
        
        # FLOPs and parameters
        total_flops = self.measure_flops_and_params(model, input_tensor, model_name)
        
        return {
            'avg_time': avg_time,
            'throughput': throughput,
            'total_flops': total_flops
        }
