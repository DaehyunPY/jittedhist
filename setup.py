from distutils.core import setup

setup(
    name='jittedhist',
    version='20180526',
    author='Daehyun You',
    author_email='daehyun.park.you@gmail.com',
    description='Simple 1d/2d histogram functions enabled to PySpark and Dask',
    long_description=open('README.md').read(),
    license='MIT',
    py_modules=['jittedhist'],
    install_requires=['numpy', 'numba', 'pyspark', 'dask']
)
