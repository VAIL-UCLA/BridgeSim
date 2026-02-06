"""
BridgeSim - Cross-simulator Closed-loop Evaluation Platform for End-to-End Autonomous Driving

A unified evaluation system for autonomous driving models using MetaDrive simulator.
"""

from setuptools import setup, find_packages

setup(
    name="bridgesim",
    version="0.1.0",
    description="Cross-simulator Closed-loop Evaluation Platform for End-to-End Autonomous Driving",
    author="BridgeSim Team",
    python_requires=">=3.8",
    packages=find_packages(include=["bridgesim", "bridgesim.*"]),
    install_requires=[
        "numpy",
        "torch",
        "opencv-python",
        "tqdm",
        "scipy",
        "shapely",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "bridgesim-eval=bridgesim.evaluation.unified_evaluator:main",
            "bridgesim-batch-eval=bridgesim.evaluation.batch_evaluator:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
