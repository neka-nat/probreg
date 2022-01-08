import os
import re
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

exec(open('probreg/version.py').read())


def _check_for_openmp():
    """Check  whether the default compiler supports OpenMP.
    This routine is adapted from pynbody // yt.
    """
    import distutils.sysconfig
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix='probreg')
    compiler = os.environ.get(
      'CC', distutils.sysconfig.get_config_var('CC'))
    if compiler is None:
        return False
    compiler = compiler.split()[0]

    # Attempt to compile a test script.
    # See http://openmp.org/wp/openmp-compilers/
    tmpfile = os.path.join(tmpdir, 'check_openmp.c')
    with open(tmpfile, 'w') as f:
        f.write('''
#include <omp.h>
#include <stdio.h>
int main() {
    #pragma omp parallel
    printf("Hello from thread %d", omp_get_thread_num());
}
''')

    try:
        with open(os.devnull, 'w') as fnull:
            exit_code = subprocess.call([compiler, '-fopenmp', '-o%s'
                                         % os.path.join(tmpdir, 'check_openmp'),
                                         tmpfile],
                                        stdout=fnull, stderr=fnull)
    except OSError:
        exit_code = 1
    finally:
        shutil.rmtree(tmpdir)

    if exit_code == 0:
        print ('Continuing your build using OpenMP...\n')
        return True
    return False


def find_eigen(hint=[]):
    """
    Find the location of the Eigen 3 include directory. This will return
    ``None`` on failure.
    """
    # List the standard locations including a user supplied hint.
    search_dirs = hint + [
        "/usr/local/include/eigen3",
        "/usr/local/homebrew/include/eigen3",
        "/opt/local/var/macports/software/eigen3",
        "/opt/local/include/eigen3",
        "/usr/include/eigen3",
        "/usr/include/local",
        "/usr/include",
    ]

    # Loop over search paths and check for the existence of the Eigen/Dense
    # header.
    for d in search_dirs:
        path = os.path.join(d, "Eigen", "Dense")
        if os.path.exists(path):
            # Determine the version.
            vf = os.path.join(d, "Eigen", "src", "Core", "util", "Macros.h")
            if not os.path.exists(vf):
                continue
            src = open(vf, "r").read()
            v1 = re.findall("#define EIGEN_WORLD_VERSION (.+)", src)
            v2 = re.findall("#define EIGEN_MAJOR_VERSION (.+)", src)
            v3 = re.findall("#define EIGEN_MINOR_VERSION (.+)", src)
            if not len(v1) or not len(v2) or not len(v3):
                continue
            v = "{0}.{1}.{2}".format(v1[0], v2[0], v3[0])
            print("Found Eigen version {0} in: {1}".format(v, d))
            return d
    return None

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        try:
            import pybind11
        except ImportError:
            if subprocess.call([sys.executable, '-m', 'pip', 'install', 'pybind11']):
                raise RuntimeError('pybind11 install failed.')
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


use_omp = _check_for_openmp()

ext_modules = [
    Extension(
        'probreg._ifgt',
        ['probreg/cc/ifgt_py.cc', 'probreg/cc/ifgt.cc', 'probreg/cc/kcenter_clustering.cc'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            find_eigen(['third_party/eigen'])
        ],
        extra_link_args=['-lgomp'] if use_omp else [],
        define_macros=[('VERSION_INFO', __version__)],
        language='c++'
    ),
    Extension(
        'probreg._math',
        ['probreg/cc/math_utils_py.cc', 'probreg/cc/math_utils.cc'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            find_eigen(['third_party/eigen'])
        ],
        extra_link_args=['-lgomp'] if use_omp else [],
        define_macros=[('VERSION_INFO', __version__)],
        language='c++'
    ),
    Extension(
        'probreg._kabsch',
        ['probreg/cc/kabsch_py.cc', 'probreg/cc/kabsch.cc'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            find_eigen(['third_party/eigen'])
        ],
        extra_link_args=['-lgomp'] if use_omp else [],
        define_macros=[('VERSION_INFO', __version__)],
        language='c++'
    ),
    Extension(
        'probreg._pt2pl',
        ['probreg/cc/point_to_plane_py.cc', 'probreg/cc/point_to_plane.cc'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            find_eigen(['third_party/eigen'])
        ],
        extra_link_args=['-lgomp'] if use_omp else [],
        define_macros=[('VERSION_INFO', __version__)],
        language='c++'
    ),
    Extension(
        'probreg._gmmtree',
        ['probreg/cc/gmmtree_py.cc', 'probreg/cc/gmmtree.cc'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            find_eigen(['third_party/eigen'])
        ],
        extra_link_args=['-lgomp'] if use_omp else [],
        define_macros=[('VERSION_INFO', __version__)],
        language='c++'
    ),
    Extension(
        'probreg._permutohedral_lattice',
        ['probreg/cc/permutohedral_lattice_py.cc', 'third_party/permutohedral/permutohedral.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            find_eigen(['third_party/eigen']),
            'third_party/permutohedral'
        ],
        define_macros=[('VERSION_INFO', __version__)],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if use_omp:
                opts.append('-fopenmp')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='probreg',
    version=__version__,
    packages=['probreg'],
    author='neka-nat',
    author_email='nekanat.stock@gmail.com',
    url='https://github.com/neka-nat/probreg',
    description='Probablistic point cloud resitration algorithms',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.2', 'open3d',
                      'six', 'transforms3d', 'scipy',
                      'scikit-learn', 'matplotlib'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
