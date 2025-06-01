from setuptools import setup, find_packages

setup(
    name="kerneloptimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "pytest>=7.0.0",
    ],
    entry_points={
        "console_scripts": [
            "kerneloptimizer=src.kernel_optimizer:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Automate CUDA kernel optimization for PyTorch using LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/KernelOptimizer",
    license="MIT",
)
