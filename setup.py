import setuptools
from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1.0",
    packages=(
        setuptools.find_packages()
        + [
            "src.tasks." + x
            for x in setuptools.find_namespace_packages(
                where="src/tasks", include=["*"]
            )
        ]
    ),
    python_requires=">=3.7",
    install_requires=[
        # List your dependencies here
        # "requests>=2.25.1",
        # "pandas>=1.2.0",
    ],
)
