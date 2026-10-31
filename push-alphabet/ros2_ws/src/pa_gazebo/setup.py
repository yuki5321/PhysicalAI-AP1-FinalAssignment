from glob import glob

from setuptools import setup

package_name = "pa_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="push-alphabet",
    maintainer_email="noreply@example.com",
    description="Push-Alphabet Gazebo Classic simulation nodes and launch.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "push_executor = pa_gazebo.push_executor:main",
            "point_push_executor = pa_gazebo.point_push_executor:main",
            "scene_manager = pa_gazebo.scene_manager:main",
            "hand_actor = pa_gazebo.hand_actor:main",
            "demo_recorder = pa_gazebo.demo_recorder:main",
            "auto_hand = pa_gazebo.demo_recorder:auto_hand_main",
        ],
    },
)
