from setuptools import setup, find_packages

setup(
    name="industrial-rl",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith(";")
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Industrial Robotics Reinforcement Learning",
    python_requires=">=3.8",
)
