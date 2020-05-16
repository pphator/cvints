import setuptools
from glob import glob


with open('README.md', 'r') as f:
    long_description = f.read()

data_files = []
directories = glob('opensets\\detection\\desktopco\\')
for directory in directories:
    files = glob(directory+'*')
    data_files.append((directory, files))

setuptools.setup(
     name='cvints',
     version='0.1',
     author="Vasily Boychuk at al",
     author_email="vasily.m.boychuk@gmail.com",
     description="A lib to solve cv tasks",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/VasilyBoychuk/cvints",
     packages=setuptools.find_packages(),
     package_data={'opensets': ['*']},
     data_files=data_files,
     include_package_data=True,
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
 )


