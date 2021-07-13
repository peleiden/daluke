from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    required = [
        l.strip() for l in f.read().splitlines() if l.strip() and not l.strip().startswith("#")
    ]
with open('optional-requirements.txt') as f:
    extras_require = [
        l.strip() for l in f.read().splitlines() if l.strip() and not l.strip().startswith("#")
    ]

# To update
# 1. Increment version below
# 2. `rm -rf dist`
# 3. `python setup.py sdist bdist_wheel`
# 4. `twine upload dist/*`
# To install locally in debug
# From root of repository, run
# `pip install --editable --upgrade .`

setup_args = dict(
    name                          = "daluke",
    version                       = "0.0.4",
    packages                      = find_packages(),
    author                        = "Søren Winkel Holm, Asger Laurits Schultz",
    author_email                  = "s18911@dtu.dk, s183912@dtu.dk",
    url                           = "https://github.com/peleiden/daLUKE",
    download_url                  = "https://pypi.org/project/daluke/",
    install_requires              = required,
    extras_require                = {"full": extras_require},
    entry_points                  = {"console_scripts": ["daluke = daluke.api.cli:main"]},
    keywords                      = [ "nlp", "ai", "pytorch", "ner" ],
    description                   = "A Danish-speaking language model with entity-aware self-attention",
    long_description              = readme,
    long_description_content_type = "text/markdown",
    license                       = "MIT License",
)

if __name__ == '__main__':
    setup(**setup_args)
