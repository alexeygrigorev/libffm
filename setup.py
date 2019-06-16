from setuptools import setup, Extension


ext = Extension(
    "ffm.libffm",
    extra_compile_args=[
          "-Wall", "-O3",
          "-std=c++0x", "-march=native",
          "-DUSESSE", "-DUSEOMP"
    ],
    extra_link_args=['-fopenmp'],
    sources=['ffm.cpp', 'timer.cpp'],
    include_dirs=['.'],
    language='c++',
)

# Please use setup_pip.py for generating and deploying pip installation
# detailed instruction in setup_pip.py
setup(name='ffm',
      version='7e8621d',
      description="LibFFM Python Package",
      long_description="LibFFM Python Package",
      install_requires=[
          'numpy',
      ],
      ext_modules=[ext],
      maintainer='',
      maintainer_email='',
      zip_safe=False,
      packages=['ffm'],
      package_data={'ffm': ['libffm.so']},
      include_package_data=True,
      license='BSD 3-clause',
      classifiers=['License :: OSI Approved :: BSD 3-clause'],
      url='https://github.com/alexeygrigorev/libffm-python'
)
