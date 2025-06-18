from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1.0",
    packages=find_packages(where="."),  # 自动发现所有包
    package_dir={"": "."},  # 指定包的根目录
)