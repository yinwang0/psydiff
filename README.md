psydiff
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



#### License (GPLv3)

psydiff - a structural comparison tool for Python

Copyright (c) 2011-2014 Yin Wang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
