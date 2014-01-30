Psydiff
=======

*a structural comparison tool for Python*

<a href="https://sourcegraph.com/github.com/yinwang0/psydiff">
<img src="https://sourcegraph.com/api/repos/github.com/yinwang0/psydiff/counters/views.png">
</a>


Psydiff is a structural differencer for Python. It parses Python into ASTs,
compare them, and then generate interactive HTML.


### Demo

<a href="http://www.yinwang.org/resources/pydiff1-pydiff2.html"><img src="http://yinwang0.files.wordpress.com/2013/07/psydiff2.gif?w=600"></a>

A demo of Psydiff's output (Psydiff diffing itself over a recent big change) can
be found here:

http://www.yinwang.org/resources/pydiff1-pydiff2.html



### Installation

1. Copy the whole directory to somewhere in your file system
2. Add the path to the system's "PATH variable"



### Usage

Just run psydiff.py from the command line:

    ./psydiff.py demos/list1.py demos/list2.py

This will generate a HTML file named list1-list2.html in the current directory.
You can then use your browser to open this file and browse around the code.

The HTML is a standalone entity (CSS styles and JavaScript embedded). You can
put it anywhere you like and still be able to view it.



### Contact

Yin Wang (yinwang0@gmail.com)



### LICENSE

Copyright (C) 2011-2013 Yin Wang

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.