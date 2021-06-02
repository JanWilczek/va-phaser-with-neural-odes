General
=======

This is the general AudioLabs LaTeX thesis template.

The main LaTeX file is _thesis.tex_.

Note: Out of the box the template does not support inclusion of .eps graphic files to allow for easy compilation on most operating systems.

Compiling
=========

Linux / Mac
-----------

To complile these LaTeX sources to a document simply type

    $ make

To get detailed output (e.g. in case of errors) type

    $ make debug

To preview this document simply type

    $ make preview

To copy the current document to your desktop and append the current date to the filename type

    $ make desktop

Windows
-------

The Windows LaTeX distribution [MiKTeX](https://en.wikipedia.org/wiki/MiKTeX) is recommended on Windows.
If [Cygwin](https://en.wikipedia.org/wiki/Cygwin) is used in addition the compilation works the same way as on Linux/Mac.

Compilation in TeXworks can be done by opening _thesis.tex_ and compiling (via the big green play button) and the "pdfLaTeX+MakeIndex+BibTeX" setting.
