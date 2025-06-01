KernelOptimizer 🚀

KernelOptimizer is an open-source tool that automates CUDA kernel optimization for PyTorch workloads using large language models (LLMs). Inspired by Stanford CRFM’s fast kernel research, it leverages natural language reasoning and branching optimization to generate high-performance GPU kernels, achieving up to 1.8x speedups over PyTorch baselines.
Whether you’re an ML engineer, researcher, or developer, KernelOptimizer simplifies GPU optimization, saving time and compute costs. Try it to boost your models’ performance!
🎥 Watch the Demo: YouTube Video (Replace with your video link)
Features

LLM-Powered Optimization: Uses LLMs (e.g., Llama) to generate creative optimization ideas and CUDA code.
Branching Optimization: Explores multiple kernel variants in parallel for maximum performance.
KernelBench Integration: Benchmarks kernels for correctness and speedup using the open-source KernelBench framework.
Easy to Use: CLI interface with example PyTorch workloads (e.g., matrix multiplication, convolution).
Nvidia GPU Support: Optimized for Nvidia GPUs (e.g., L40S, A100) with FP32 precision.

Installation

Clone the Repository:git clone https://github.com/yourusername/KernelOptimizer.git
cd KernelOptimizer


Install Dependencies:pip install -r requirements.txt


Install KernelBench:Follow the KernelBench documentation to set up the benchmarking framework.
Set Up CUDA Toolkit:Ensure the CUDA Toolkit (11.8 or later) is installed for your Nvidia GPU.
Optional: Install LLM:Use a local model like Llama via Hugging Face Transformers or configure an API (e.g., OpenAI).

Usage
Optimize a PyTorch workload with a single command:
python -m src.kernel_optimizer --input examples/matrix_multiply.py --problem-size 1024 --iterations 5 --output optimized_kernel.cu


Input: Path to your PyTorch code (e.g., examples/matrix_multiply.py).
Problem Size: Size of the workload (e.g., 1024 for a 1024x1024 matrix).
Iterations: Number of optimization rounds (default: 5).
Output: Path to save the optimized CUDA kernel.

Example Output:
2025-05-31 22:43:15 - INFO - Saved best kernel to optimized_kernel.cu: {'fast_p': 0.85, 'speedup': 85.0, 'correct': True}

Examples

Matrix Multiplication: Optimize a matrix multiply kernel (examples/matrix_multiply.py) for a 1.8x speedup.
Convolution: Generate a high-performance kernel for a 2D convolution (examples/convolution.py).

See the examples/ folder for more workloads.
Contributing
We welcome contributions to make KernelOptimizer better! To contribute:

Fork the repository.
Create a branch: git checkout -b feature/your-feature.
Make changes and test: pytest tests/.
Submit a pull request.

Check CONTRIBUTING.md for details. Ideas:

Add support for new PyTorch operators.
Implement AMD GPU support via ROCm.
Improve LLM integration.

Community

Discuss: Join our GitHub Discussions to share ideas or ask questions.
Report Issues: Use GitHub Issues for bugs or feature requests.
Connect: Follow us on X for updates and demos.

Support the Project
Loved the speedup? Consider:

⭐ Star the repo to show your support!
☕ Buy me a coffee to fuel development: Buy Me a Coffee (Replace with your link).
💼 Hire me for custom kernel optimization: Contact (Replace with your email).

License
KernelOptimizer is licensed under the MIT License.
Acknowledgments

Inspired by Stanford CRFM’s fast kernel research.
Built with KernelBench and Hugging Face Transformers.


Star KernelOptimizer today and supercharge your ML workloads! 🚀

