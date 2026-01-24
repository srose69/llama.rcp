#!/usr/bin/env python3
"""
Setup script for llamarcp wrapper.
"""

from setuptools import setup, find_packages

setup(
    name="llamarcp",
    version="0.3.16",
    description="Python wrapper for llama.rcp",
    packages=find_packages(include=["llamarcp", "llamarcp.*"]),
    package_data={
        "llamarcp": ["lib/*.so*", "py.typed"],
    },
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "server": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "sse-starlette>=1.6.0",
            "pydantic-settings>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llamarcp=llamarcp.__main__:main",
        ],
    },
)
