from setuptools import setup

package_name = "pa_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    package_data={package_name: ["scene.yaml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="push-alphabet",
    maintainer_email="noreply@example.com",
    description="Shared geometry, push model and planner for Push-Alphabet.",
    license="MIT",
    tests_require=["pytest"],
)
