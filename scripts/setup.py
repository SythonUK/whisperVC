from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
	cmdclass = {'build_ext' : build_ext},
	ext_modules = [Extension('chol', ['chol.pyx']), Extension('hts_mlpg', ['hts_mlpg.pyx'])]
)
