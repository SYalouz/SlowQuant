
Installation and Running
========================

How to install SlowQaunt, and run the program.

Installation Ubuntu
-------------------

This installation guide is made for Ubuntu. 

Python 3.6 can be installed with conda running the following command lines:

::
  
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

Then run:

::
  
  bash Miniconda3-latest-Linux-x86_64.sh

The python packages can now be installed by:

::
  
  conda install numpy scipy numba

The SlowQuant program can be downloaded by:

::
  
  git clone https://github.com/Melisius/SlowQuant.git

To test the installation install pytest:

::
  
  conda install pytest

And then run:

::
  
  pytest tests.py

All of the tests should succeed.

Installation Windows 10
-----------------------

For Windows 10, an Ubuntu can be installed following the guide in the following link:

https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/

Here after all the above steps can be followed.

Running SlowQuant
-----------------

SOMETHING HERE

Compiling documentation
-----------------------

The documentation is compiled using sphinx:

::
  
  conda install sphinx sphinx_rtd_theme

To make the equations LaTeX is needed, and can be installed by:

::
  
  sudo apt-get install texlive-full

The documentation is now compiled by:

::
  
  sphinx-build data/documentation docs
  
