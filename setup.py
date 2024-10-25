import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as fh:
    long_description = fh.read()

package_name = "cppflow"

setup(
    name=package_name,
    version="0.1.0",
    description="",
    author="Jeremy Morgan, David Millard",
    author_email="jsmorgan6@gmail.com, dmillard@gmail.com",
    license="MIT",
    long_description_content_type="text/markdown",
    long_description=long_description,
    python_requires=">=3.8.0,<3.11",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.7.0",
        "psutil==5.9.8",
    ],
    data_files=[
        ('share/' + package_name, ['package.xml', 'README.md']),  # Installs package.xml and README.md to /workspaces/bdai/_build/install/cppflow/share/cppflow
    ],
    entry_points={
        'console_scripts': [
            'subscriber = cppflow.scripts.ros2_subscriber:main',
        ],
    },
    dependency_links=[
        "git+https://github.com/jstmn/jrl.git@ebb937d2189369cce5eace9b1f83afd39df32176#egg=jrl",
        "git+https://github.com/jstmn/ikflow.git@b45c92b4147400bdb45dbabfd7169e62c9c0d187#egg=ikflow",
    ],
    extras_require={
        "dev": [
            "black==23.1.0",
            "pylint==2.16.2",
            "matplotlib==3.7.0",
            "pandas==1.5.3",
            "tabulate==0.9.0",
            "pytest==8.2.0",
        ]
    },
)
