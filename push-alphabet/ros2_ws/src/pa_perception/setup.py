from glob import glob

from setuptools import setup

package_name = "pa_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        # CNN model (pa_perception.cnn_detector); absent until trained -> empty glob.
        ("share/" + package_name + "/models", glob("models/*.onnx") + glob("models/*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="push-alphabet",
    maintainer_email="noreply@example.com",
    description="Overhead-camera letter detection and pose estimation for Push-Alphabet.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "letter_detector = pa_perception.letter_detector_node:main",
        ],
    },
)
