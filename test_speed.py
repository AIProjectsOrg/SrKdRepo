import torch
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from models.team07_MicroSR.model import MicroSR
from models.team07_MicroSR.io import forward
from basicsr.archs.swinir_arch import SwinIR


def benchmark_model(model, input_tensor, device, model_name="Model", warmup=10, runs=50):
    """Benchmark inference time and throughput"""
    model.eval().to(device)
    input_tensor = input_tensor.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / runs
    throughput = 1.0 / avg_time
    print(f"\n==== {model_name} Benchmark ====")
    print(f"Inference Time: {avg_time * 1000:.2f} ms/image")
    print(f"Throughput: {throughput:.2f} images/sec")
    return avg_time, throughput


def measure_flops(model, input_tensor, model_name="Model"):
    """Estimate FLOPs using fvcore"""
    model.eval()
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"\n==== {model_name} FLOPs & Params ====")
    print(parameter_count_table(model))
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    return flops.total()


class WrappedTeacher(torch.nn.Module):
    """Wrap teacher model to match forward(x) API"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return forward(x, self.model, tile=None)


def load_teacher(device):
    """Load the teacher (MicroSR) model"""
    model_dir = "/content/NTIRE2025_ImageSR_x4/model_zoo/team07_MicroSR/MicroSR_X4"
    teacher = MicroSR(
        upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
        squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
        depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        gc=32, mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
    )
    checkpoint = torch.load(model_dir, map_location=device, weights_only=False)
    teacher.load_state_dict(checkpoint['params'], strict=True)
    return WrappedTeacher(teacher).to(device)


def load_student(device):
    """Load the student (SwinIR) model from checkpoint"""
    student = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[2, 2, 2, 2],            # fewer transformer layers
        embed_dim=60,                   # smaller embedding size
        num_heads=[2, 2, 2, 2],         # fewer attention heads
        mlp_ratio=1.5,                  # reduced MLP size
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    ckpt = torch.load('best_student_model.pth', map_location=device, weights_only=False)
    student.load_state_dict(ckpt['model_state_dict'], strict=True)
    return student.to(device)


def main():
    # ==== HARD-CODED SETTINGS ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 64  # Input image size (HxW)
    warmup_iters = 10
    timed_iters = 50

    print(f"Running benchmark on: {device}")
    print(f"Input image size: {img_size}x{img_size}")
    print(f"Warm-up: {warmup_iters}, Timed runs: {timed_iters}")

    # Dummy input tensor
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # Load models
    teacher_model = load_teacher(device)
    student_model = load_student(device)

    # Benchmark teacher
    benchmark_model(teacher_model, dummy_input, device, "Teacher", warmup_iters, timed_iters)
    measure_flops(teacher_model, dummy_input, "Teacher")

    # Benchmark student
    benchmark_model(student_model, dummy_input, device, "Student", warmup_iters, timed_iters)
    measure_flops(student_model, dummy_input, "Student")


if __name__ == "__main__":
    main()
