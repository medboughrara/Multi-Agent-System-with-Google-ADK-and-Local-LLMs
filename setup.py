"""
Setup script for the ADK package.
"""

from setuptools import setup, find_packages

setup(
    name="adk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "psutil>=5.9.0",
        "gputil>=1.4.0",
        "pynvml>=11.0.0",
    ],
    author="Local ADK Development",
    description="Local implementation of Google's Agent Development Kit (ADK)",
    python_requires=">=3.8",
)
