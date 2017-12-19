from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("overlap_NEW",
               ["overlap_NEW.pyx"],
               libraries=["m"],
               extra_compile_args = ["-ffast-math"])]

setup(
        name = "overlap_NEW",
        cmdclass = {"build_ext": build_ext},
        ext_modules = ext_modules)
