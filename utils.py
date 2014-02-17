#-------------------------------------------------------------
#            tests and operations on AST nodes
#-------------------------------------------------------------
import os
import sys
import re
import cProfile

from ast import *
from parameters import *


# get list of fields from a node
def node_fields(node):
    ret = []
    for field in node._fields:
        if field != 'ctx' and hasattr(node, field):
            ret.append(getattr(node, field))
    return ret



# get full source text where the node is from
def node_source(node):
    if hasattr(node, 'node_source'):
        return node.node_source
    else:
        return None



# utility for getting exact source code part of the node
def src(node):
    return node.node_source[node.node_start : node.node_end]



def node_start(node):
    if (hasattr(node, 'node_start')):
        return node.node_start
    else:
        return 0



def node_end(node):
    if hasattr(node, 'node_end'):
        return node.node_end
    else:
        return None


def is_atom(x):
    return type(x) in [int, str, bool, float]



def is_def(node):
    return isinstance(node, FunctionDef) or isinstance(node, ClassDef)



# whether a node is a "frame" which can contain others and be
# labeled
def is_frame(node):
    return type(node) in [ClassDef, FunctionDef, Import, ImportFrom]



def is_empty_container(node):
    if isinstance(node, List) and node.elts == []:
        return True
    if isinstance(node, Tuple) and node.elts == []:
        return True
    if isinstance(node, Dict) and node.keys == []:
        return True

    return False


def same_def(node1, node2):
    if isinstance(node1, FunctionDef) and isinstance(node2, FunctionDef):
        return node1.name == node2.name
    elif isinstance(node1, ClassDef) and isinstance(node2, ClassDef):
        return node1.name == node2.name
    else:
        return False


def different_def(node1, node2):
    if is_def(node1) and is_def(node2):
        return node1.name != node2.name
    return False


# decide whether it is reasonable to consider two nodes to be
# moves of each other
def can_move(node1, node2, cost):
    return (same_def(node1, node2) or
            cost <= (node_size(node1) + node_size(node2)) * MOVE_RATIO)


# whether the node is considered deleted or inserted because
# the other party matches a substructure of it.
def node_framed(node, changes):
    for c in changes:
        if (c.is_frame and (node == c.orig or node == c.cur)):
            return True
    return False



# helper for turning nested if statements into sequences,
# otherwise we will be trapped in the nested structure and find
# too many differences
def serialize_if(node):
    if isinstance(node, If):
        if not hasattr(node, 'node_end'):
            print("has no end:", node)

        newif = If(node.test, node.body, [])
        newif.lineno = node.lineno
        newif.col_offset = node.col_offset
        newif.node_start = node.node_start
        newif.node_end = node.node_end
        newif.node_source = node.node_source
        newif.fileName = node.fileName
        return [newif] + serialize_if(node.orelse)
    elif isinstance(node, list):
        ret = []
        for n in node:
            ret += serialize_if(n)
        return ret
    else:
        return [node]


def node_name(node):
    if isinstance(node, Name):
        return node.id
    elif isinstance(node, FunctionDef) or isinstance(node, ClassDef):
        return node.name
    else:
        return None


def attr_to_str(node):
    if isinstance(node, Attribute):
        vName = attr_to_str(node.value)
        if vName != None:
            return vName + "." + node.attr
        else:
            return None
    elif isinstance(node, Name):
        return node.id
    else:
        return None


### utility for counting size of terms
def node_size(node, test=False):

    if not test and hasattr(node, 'node_size'):
        ret = node.node_size

    elif isinstance(node, list):
        ret = sum(map(lambda x: node_size(x, test), node))

    elif is_atom(node):
        ret = 1

    elif isinstance(node, Name):
        ret = 1

    elif isinstance(node, Num):
        ret = 1

    elif isinstance(node, Str):
        ret = 1

    elif isinstance(node, Expr):
        ret = node_size(node.value, test)

    elif isinstance(node, AST):
        ret = 1 + sum(map(lambda x: node_size(x, test), node_fields(node)))

    else:
        ret = 0

    if test:
        print("node:", node, "size=", ret)

    if isinstance(node, AST):
        node.node_size = ret

    return ret



#-------------------------------------------------------------
# utilities
#-------------------------------------------------------------
def debug(*args):
    if DEBUG:
        print(args)


def dot():
    sys.stdout.write('.')
    sys.stdout.flush()


def is_alpha(c):
    return (c == '_'
            or ('0' <= c <= '9')
            or ('a' <= c <= 'z')
            or ('A' <= c <= 'Z'))


def div(m, n):
    if n == 0:
        return m
    else:
        return m/float(n)


# for debugging
def ps(s):
    v = parse(s).body[0]
    if isinstance(v, Expr):
        return v.value
    else:
        return v


def sz(s):
    return node_size(parse(s), True) - 1


def dp(s):
    return dump(parse(s))


def run(name, closure=True, debug=False):
    fullname1 = name + '1.py'
    fullname2 = name + '2.py'

    global DEBUG
    olddebug = DEBUG
    DEBUG = debug

    diff(fullname1, fullname2, closure)

    DEBUG = olddebug


def demo():
    run('demo')


def go():
    run('heavy')


def pf():
    cProfile.run("run('heavy')", sort="cumulative")




#------------------------ file system support -----------------------

def base_name(filename):
    try:
        start = filename.rindex('/') + 1
    except ValueError:
        start = 0

    try:
        end = filename.rindex('.py')
    except ValueError:
        end = 0
    return filename[start:end]


## file system support
def parse_file(filename):
    f = open(filename, 'r');
    lines = f.read()
    ast = parse(lines)
    improve_ast(ast, lines, filename, 'left')
    return ast


def get_install_path():
    exec_name = os.path.abspath(__file__)
    path = exec_name.rindex(os.sep) + 1
    return exec_name[:path]


def lfilter(f, ls):
    return list(filter(f, ls))

