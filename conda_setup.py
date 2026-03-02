import os
from shutil import which

import setuptools


def install_requirements_yml(path):
    installer = 'mamba' if which('mamba') else 'conda'
    if exitcode := os.system(f"{installer} env update -f {path}"):
        raise OSError(f"conda installation failed with exitcode {exitcode}")


def setup(*args, requirements_yml=None, **kwargs):
    from setuptools.command import install as _install, develop as _develop

    if requirements_yml:

        class install(_install.install):
            user_options = _install.install.user_options + [
                ("no-deps", "N", "don't install dependencies")
                ]

            def initialize_options(self):
                super().initialize_options()
                self.no_deps = None

            def run(self):
                if not self.no_deps:
                    install_requirements_yml(requirements_yml)
                super().run()

        class develop(_develop.develop):
            def run(self):
                if not self.no_deps:
                    install_requirements_yml(requirements_yml)
                super().run()

        kwargs.setdefault("cmdclass", {}).update(
            {"install": install, "develop": develop}
            )
    setuptools.setup(*args, **kwargs)
