"""
Modular speed testing script for benchmarking model performance.
"""
import torch

from config import Config
from models.model_loader import ModelLoader
from utils.benchmark import ModelBenchmark


def main():
    """Main benchmarking function"""
    # Setup
    config = Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on: {device}")
    print(f"Input image size: {config.BENCHMARK_INPUT_SIZE}x{config.BENCHMARK_INPUT_SIZE}")
    print(f"Warm-up: {config.BENCHMARK_WARMUP}, Timed runs: {config.BENCHMARK_RUNS}")
    
    # Initialize benchmark utility
    benchmark = ModelBenchmark(device)
    
    # Create dummy input tensor
    dummy_input = torch.randn(
        1, 3, config.BENCHMARK_INPUT_SIZE, config.BENCHMARK_INPUT_SIZE
    ).to(device)
    
    # Load models
    print("\nLoading models...")
    teacher_model = ModelLoader.load_teacher(
        device, 
        config.TEACHER_MODEL_PATH, 
        wrapped=True  # Wrap for standardized forward API
    )
    
    student_model = ModelLoader.load_student(device, config.STUDENT_CHECKPOINT_PATH)
    
    # Benchmark teacher model
    print("\n" + "="*60)
    print("BENCHMARKING TEACHER MODEL")
    print("="*60)
    teacher_results = benchmark.full_benchmark(
        teacher_model,
        dummy_input,
        "Teacher (MicroSR)",
        config.BENCHMARK_WARMUP,
        config.BENCHMARK_RUNS
    )
    
    # Benchmark student model
    print("\n" + "="*60)
    print("BENCHMARKING STUDENT MODEL") 
    print("="*60)
    student_results = benchmark.full_benchmark(
        student_model,
        dummy_input,
        "Student (SwinIR)",
        config.BENCHMARK_WARMUP,
        config.BENCHMARK_RUNS
    )
    
    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    if teacher_results['avg_time'] and student_results['avg_time']:
        speedup = teacher_results['avg_time'] / student_results['avg_time']
        print(f"Speed improvement: {speedup:.2f}x faster")
    
    if teacher_results['total_flops'] and student_results['total_flops']:
        flop_reduction = (teacher_results['total_flops'] - student_results['total_flops']) / teacher_results['total_flops'] * 100
        print(f"FLOPs reduction: {flop_reduction:.1f}%")
    
    print(f"\nTeacher - Time: {teacher_results['avg_time']*1000:.2f}ms, "
          f"Throughput: {teacher_results['throughput']:.2f} img/s")
    print(f"Student - Time: {student_results['avg_time']*1000:.2f}ms, "
          f"Throughput: {student_results['throughput']:.2f} img/s")


if __name__ == "__main__":
    main()
