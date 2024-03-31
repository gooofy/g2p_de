from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'g2p_de',
  packages = ['g2p_de'], # this must be the same as the name above
  version = '1.1.0',
  description = 'A Simple Python Module for German Grapheme To Phoneme Conversion',
  long_description=long_description,
  author = 'Günter Bartsch, Kyubyong Park, Jongseok Kim',
  author_email = 'guenter@zamia.org',
  url = 'https://github.com/gooofy/g2p_de',
  download_url = 'https://github.com/gooofy/g2p_de/archive/1.0.2.tar.gz',
  keywords = ['g2p','g2p_de'],
  classifiers = [],
  install_requires = [
    'numpy>=1.13.1',
    'nltk>=3.2.4',
    'num2words>=0.5.13',
  ],
  license='Apache Software License',
  include_package_data=True
)

