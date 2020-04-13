import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()


setuptools.setup(
     name='cvints',
     version='0.1',
     author="Vasily Boychuk at al",
     author_email="vasily.m.boychuk@gmail.com",
     description="A lib to solve cv tasks",
     long_description=long_description,
  long_description_content_type="text/markdown",
     url="https://github.com/VasilyBoychuk/cvints",
     packages=setuptools.find_packages(exclude=['examples']),
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
 )


