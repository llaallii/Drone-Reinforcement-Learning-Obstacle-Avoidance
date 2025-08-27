from setuptools import setup, find_packages

setup(
    name="drone_forest_navigation_rl",
    version="0.1.0",
    description="RL for drone navigation in forest environments",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[line.strip() for line in open("requirements.txt") if line.strip() and not line.startswith("#")],
)
