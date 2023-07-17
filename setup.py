from setuptools import setup


setup(
    use_scm_version={
        "write_to": "kalman_reconstruction/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
        "local_scheme": "node-and-date",
    },
    description="Data-driven Reconstruction of Partially Observed Dynamical Systems using Kalman algorithms and an iterative procedure.",
    author="Nils Niebaum",
    license="",
)
