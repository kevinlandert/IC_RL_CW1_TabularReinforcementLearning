import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="project_template",
    version="0.0.1",
    author="Kevin Landert",
    author_email="kevin.landert@gmail.com",
    description="Template organising the file structure for projects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="add github url here",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)