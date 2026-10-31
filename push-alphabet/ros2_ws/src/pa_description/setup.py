import os
from glob import glob

from setuptools import setup

package_name = "pa_description"


def tree(src):
    """data_files entries that install `src/**` under share/<pkg>/<src>, keeping the layout."""
    out = {}
    for path in glob(os.path.join(src, "**", "*"), recursive=True):
        if os.path.isfile(path):
            out.setdefault(os.path.join("share", package_name, os.path.dirname(path)), []).append(path)
    return list(out.items())


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ] + tree("models"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="push-alphabet",
    maintainer_email="noreply@example.com",
    description="Push-Alphabet robot description generated from scene.yaml.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["gen_urdf = pa_description.urdf_gen:main"]},
)
