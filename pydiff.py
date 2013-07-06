#!/usr/bin/env python

import sys
import re
import cProfile

from ast import *
from lists import *



#-------------------------------------------------------------
# global parameters
#-------------------------------------------------------------

DEBUG = False
# sys.setrecursionlimit(10000)


MOVE_RATIO     = 0.2
MOVE_SIZE      = 10
MOVE_ROUND     = 5

FRAME_DEPTH    = 1
FRAME_SIZE     = 20

NAME_PENALTY   = 1
IF_PENALTY     = 1
ASSIGN_PENALTY = 1


#-------------------------------------------------------------
# utilities
#-------------------------------------------------------------
def debug(*args):
    if DEBUG:
        print args


def dot():
    sys.stdout.write('.')


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




#-------------------------------------------------------------
#            tests and operations on AST nodes
#-------------------------------------------------------------

# get list of fields from a node
def node_fields(node):
    ret = []
    for field in node._fields:
        if field <> 'ctx' and hasattr(node, field):
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
    return node.node_end



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
        return node1.name <> node2.name
    return False


# decide whether it is reasonable to consider two nodes to be
# moves of each other
def can_move(node1, node2, c):
    return (same_def(node1, node2) or
            c <= (node_size(node1) + node_size(node2)) * MOVE_RATIO)


# whether the node is considered deleted or inserted because
# the other party matches a substructure of it.
def nodeFramed(node, changes):
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
            print "has no end:", node

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


def attr2str(node):
    if isinstance(node, Attribute):
        vName = attr2str(node.value)
        if vName <> None:
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
        print "node:", node, "size=", ret

    if isinstance(node, AST):
        node.node_size = ret

    return ret




#------------------------------- types ------------------------------
# global storage of running stats
class Stat:
    def __init__(self):
        pass

stat = Stat()



# The difference between nodes are stored as a Change structure.
class Change:
    def __init__(self, orig, cur, cost, is_frame=False):
        self.orig = orig
        self.cur = cur
        if orig == None:
            self.cost = node_size(cur)
        elif cur == None:
            self.cost = node_size(orig)
        elif cost == 'all':
            self.cost = node_size(orig) + node_size(cur)
        else:
            self.cost = cost
        self.is_frame = is_frame
    def __repr__(self):
        fr = "F" if self.is_frame else "-"
        def hole(x):
            if x == None:
                return "[]"
            else:
                return x
        return ("(C:" + str(hole(self.orig)) + ":" + str(hole(self.cur))
                + ":" + str(self.cost) + ":" + str(self.similarity())
                + ":" + fr + ")")
    def similarity(self):
        total = node_size(self.orig) + node_size(self.cur)
        return 1 - div(self.cost, total)



# Three major kinds of changes:
# * modification
# * deletion
# *insertion
def modify_node(node1, node2, cost):
    return loner(Change(node1, node2, cost))

def del_node(node):
    return loner(Change(node, None, node_size(node)))

def ins_node(node):
    return loner(Change(None, node, node_size(node)))



# general cache table for acceleration
class Cache:
    def __init__(self):
        self.table = {}
    def __repr__(self):
        return "Cache:" + str(self.table)
    def __len__(self):
        return len(self.table)
    def put(self, key, value):
        self.table[key] = value
    def get(self, key):
        if self.table.has_key(key):
            return self.table[key]
        else:
            return None



# 2-D array table for memoization of dynamic programming
def create_table(x, y):
    table = []
    for i in range(x+1):
        table.append([None] * (y+1))
    return table

def tableLookup(t, x, y):
    return t[x][y]

def tablePut(t, x, y, v):
    t[x][y] = v





#-------------------------------------------------------------
#                  string distance function
#-------------------------------------------------------------

### diff cache for AST nodes
str_dist_cache = Cache()
def clear_str_dist_cache():
    global str_dist_cache
    str_dist_cache = Cache()


### string distance function
def str_dist(s1, s2):
    cached = str_dist_cache.get((s1, s2))
    if cached <> None:
        return cached

    if len(s1) > 100 or len(s2) > 100:
        if s1 <> s2:
            return 2.0
        else:
            return 0

    table = create_table(len(s1), len(s2))
    d = dist1(table, s1, s2)
    ret = div(2*d, len(s1) + len(s2))

    str_dist_cache.put((s1, s2), ret)
    return ret


# the main dynamic programming part
# similar to the structure of diff_list
def dist1(table, s1, s2):
    def memo(v):
        tablePut(table, len(s1), len(s2), v)
        return v

    cached = tableLookup(table, len(s1), len(s2))
    if (cached <> None):
        return cached

    if s1 == '':
        return memo(len(s2))
    elif s2 == '':
        return memo(len(s1))
    else:
        if s1[0] == s2[0]:
            d0 = 0
        elif s1[0].lower() == s2[0].lower():
            d0 = 1
        else:
            d0 = 2

        d0 = d0 + dist1(table, s1[1:], s2[1:])
        d1 = 1 + dist1(table, s1[1:], s2)
        d2 = 1 + dist1(table, s1, s2[1:])
        return memo(min(d0, d1, d2))




#-------------------------------------------------------------
#                        diff of nodes
#-------------------------------------------------------------

stat.diff_count = 0
def diff_node(node1, node2, env1, env2, depth, move):

    # try substructural diff
    def trysub((changes, cost)):
        if not move:
            return (changes, cost)
        elif can_move(node1, node2, cost):
            return (changes, cost)
        else:
            mc1 = diff_subnode(node1, node2, env1, env2, depth, move)
            if mc1 <> None:
                return mc1
            else:
                return (changes, cost)

    if isinstance(node1, list) and not isinstance(node2, list):
        return diff_node(node1, [node2], env1, env2, depth, move)

    if not isinstance(node1, list) and isinstance(node2, list):
        return diff_node([node1], node2, env1, env2, depth, move)

    if (isinstance(node1, list) and isinstance(node2, list)):
        node1 = serialize_if(node1)
        node2 = serialize_if(node2)
        table = create_table(len(node1), len(node2))
        return diff_list(table, node1, node2, env1, env2, 0, move)

    # statistics
    stat.diff_count += 1
    if stat.diff_count % 1000 == 0:
        dot()

    if node1 == node2:
        return (modify_node(node1, node2, 0), 0)

    if isinstance(node1, Num) and isinstance(node2, Num):
        if node1.n == node2.n:
            return (modify_node(node1, node2, 0), 0)
        else:
            return (modify_node(node1, node2, 1), 1)

    if isinstance(node1, Str) and isinstance(node2, Str):
        cost = str_dist(node1.s, node2.s)
        return (modify_node(node1, node2, cost), cost)

    if (isinstance(node1, Name) and isinstance(node2, Name)):
        v1 = lookup(node1.id, env1)
        v2 = lookup(node2.id, env2)
        if v1 <> v2 or (v1 == None and v2 == None):
            cost = str_dist(node1.id, node2.id)
            return (modify_node(node1, node2, cost), cost)
        else:                           # same variable
            return (modify_node(node1, node2, 0), 0)

    if (isinstance(node1, Attribute) and isinstance(node2, Name) or
        isinstance(node1, Name) and isinstance(node2, Attribute) or
        isinstance(node1, Attribute) and isinstance(node2, Attribute)):
        s1 = attr2str(node1)
        s2 = attr2str(node2)
        if s1 <> None and s2 <> None:
            cost = str_dist(s1, s2)
            return (modify_node(node1, node2, cost), cost)
        # else fall through for things like f(x).y vs x.y

    if isinstance(node1, Module) and isinstance(node2, Module):
        return diff_node(node1.body, node2.body, env1, env2, depth, move)

    # other AST nodes
    if (isinstance(node1, AST) and isinstance(node2, AST) and
        type(node1) == type(node2)):

        fs1 = node_fields(node1)
        fs2 = node_fields(node2)
        changes, cost = nil, 0

        for i in xrange(len(fs1)):
            (m, c) = diff_node(fs1[i], fs2[i], env1, env2, depth, move)
            changes = append(m, changes)
            cost += c

        return trysub((changes, cost))

    if (type(node1) == type(node2) and
             is_empty_container(node1) and is_empty_container(node2)):
        return (modify_node(node1, node2, 0), 0)

    # all unmatched types and unequal values
    return trysub((append(del_node(node1), ins_node(node2)),
                   node_size(node1) + node_size(node2)))



########################## diff of a list ##########################

# diff_list is the main part of dynamic programming

def diff_list(table, ls1, ls2, env1, env2, depth, move):

    def memo(v):
        tablePut(table, len(ls1), len(ls2), v)
        return v

    def guess(table, ls1, ls2, env1, env2):
        (m0, c0) = diff_node(ls1[0], ls2[0], env1, env2, depth, move)
        (m1, c1) = diff_list(table, ls1[1:], ls2[1:], env1, env2, depth, move)
        cost1 = c1 + c0

        if ((is_frame(ls1[0]) and
             is_frame(ls2[0]) and
             not nodeFramed(ls1[0], m0) and
             not nodeFramed(ls2[0], m0))):
            frameChange = modify_node(ls1[0], ls2[0], c0)
        else:
            frameChange = nil

        # short cut 1 (func and classes with same names)
        if can_move(ls1[0], ls2[0], c0):
            return (append(frameChange, m0, m1), cost1)

        else:  # do more work
            (m2, c2) = diff_list(table, ls1[1:], ls2, env1, env2, depth, move)
            (m3, c3) = diff_list(table, ls1, ls2[1:], env1, env2, depth, move)
            cost2 = c2 + node_size(ls1[0])
            cost3 = c3 + node_size(ls2[0])

            if (not different_def(ls1[0], ls2[0]) and
                cost1 <= cost2 and cost1 <= cost3):
                return (append(frameChange, m0, m1), cost1)
            elif (cost2 <= cost3):
                return (append(del_node(ls1[0]), m2), cost2)
            else:
                return (append(ins_node(ls2[0]), m3), cost3)

    # cache look up
    cached = tableLookup(table, len(ls1), len(ls2))
    if (cached <> None):
        return cached

    if (ls1 == [] and ls2 == []):
        return memo((nil, 0))

    elif (ls1 <> [] and ls2 <> []):
        return memo(guess(table, ls1, ls2, env1, env2))

    elif ls1 == []:
        d = nil
        for n in ls2:
            d = append(ins_node(n), d)
        return memo((d, node_size(ls2)))

    else: # ls2 == []:
        d = nil
        for n in ls1:
            d = append(del_node(n), d)
        return memo((d, node_size(ls1)))




###################### diff into a subnode #######################

# Subnode diff is only used in the moving phase. There is no
# need to compare the substructure of two nodes in the first
# run, because they will be reconsidered if we just consider
# them to be complete deletion and insertions.

def diff_subnode(node1, node2, env1, env2, depth, move):

    if (depth >= FRAME_DEPTH or
        node_size(node1) < FRAME_SIZE or
        node_size(node2) < FRAME_SIZE):
        return None

    if isinstance(node1, AST) and isinstance(node2, AST):

        if node_size(node1) == node_size(node2):
            return None

        if isinstance(node1, Expr):
            node1 = node1.value

        if isinstance(node2, Expr):
            node2 = node2.value

        if (node_size(node1) < node_size(node2)):
            for f in node_fields(node2):
                (m0, c0) = diff_node(node1, f, env1, env2, depth+1, move)
                if can_move(node1, f, c0):
                    if not isinstance(f, list):
                        m1 = modify_node(node1, f, c0)
                    else:
                        m1 = nil
                    framecost = node_size(node2) - node_size(node1)
                    m2 = loner(Change(None, node2, framecost, True))
                    return (append(m2, m1, m0), c0 + framecost)

        if (node_size(node1) > node_size(node2)):
            for f in node_fields(node1):
                (m0, c0) = diff_node(f, node2, env1, env2, depth+1, move)
                if can_move(f, node2, c0):
                    framecost = node_size(node1) - node_size(node2)
                    if not isinstance(f, list):
                        m1 = modify_node(f, node2, c0)
                    else:
                        m1 = nil
                    m2 = loner(Change(node1, None, framecost, True))
                    return (append(m2, m1, m0), c0 + framecost)

    return None





##########################################################################
##                          move detection
##########################################################################
def move_candidate(node):
    return (is_def(node) or node_size(node) >= MOVE_SIZE)


stat.move_count = 0
stat.move_savings = 0
def get_moves(ds, round=0):

    dels = pylist(filterlist(lambda p: (p.cur == None and
                                        move_candidate(p.orig) and
                                        not p.is_frame),
                             ds))
    adds = pylist(filterlist(lambda p: (p.orig == None and
                                        move_candidate(p.cur) and
                                        not p.is_frame),
                             ds))

    # print "dels=", dels
    # print "adds=", adds

    matched = []
    newChanges, total = nil, 0

    print("\n[get_moves #%d] %d * %d = %d pairs of nodes to consider ..."
          % (round, len(dels), len(adds), len(dels) * len(adds)))

    for d0 in dels:
        for a0 in adds:
            (node1, node2) = (d0.orig, a0.cur)
            (changes, cost) = diff_node(node1, node2, nil, nil, 0, True)
            nterms = node_size(node1) + node_size(node2)

            if (can_move(node1, node2, cost) or
                nodeFramed(node1, changes) or
                nodeFramed(node2, changes)):

                matched.append(d0)
                matched.append(a0)
                adds.remove(a0)
                newChanges = append(changes, newChanges)
                total += cost

                if (not nodeFramed(node1, changes) and
                    not nodeFramed(node2, changes) and
                    is_def(node1) and is_def(node2)):
                    newChanges = append(modify_node(node1, node2, cost),
                                        newChanges)

                stat.move_savings += nterms
                stat.move_count +=1
                if stat.move_count % 1000 == 0:
                    dot()

                break

    print("\n\t%d matched pairs found with %d new changes."
          % (len(pylist(matched)), len(pylist(newChanges))))

    # print "matches=", matched
    # print "newChanges=", newChanges

    return (matched, newChanges, total)



# Get moves repeatedly because new moves may introduce new
# deletions and insertions.

def closure(res):
    (changes, cost) = res
    matched = None
    moveround = 1

    while moveround <= MOVE_ROUND and matched <> []:
        (matched, newChanges, c) = get_moves(changes, moveround)
        moveround += 1
        # print "matched:", matched
        # print "changes:", changes
        changes = filterlist(lambda c: c not in matched, changes)
        changes = append(newChanges, changes)
        savings = sum(map(lambda p: node_size(p.orig) + node_size(p.cur), matched))
        cost = cost + c - savings
    return (changes, cost)





#-------------------------------------------------------------
#                   improvements to the AST
#-------------------------------------------------------------

allNodes1 = set()
allNodes2 = set()

def improve_node(node, s, idxmap, filename, side):

    if isinstance(node, list):
        for n in node:
            improve_node(n, s, idxmap, filename, side)

    elif isinstance(node, AST):

        if side == 'left':
            allNodes1.add(node)
        else:
            allNodes2.add(node)

        find_node_start(node, s, idxmap)
        find_node_end(node, s, idxmap)
        add_missing_names(node, s, idxmap)

        node.node_source = s
        node.fileName = filename

        for f in node_fields(node):
            improve_node(f, s, idxmap, filename, side)



def improve_ast(node, s, filename, side):
    idxmap = build_index_map(s)
    improve_node(node, s, idxmap, filename, side)




#-------------------------------------------------------------
#            finding start and end index of nodes
#-------------------------------------------------------------

def find_node_start(node, s, idxmap):

    if hasattr(node, 'node_start'):
        return node.node_start

    elif isinstance(node, list):
        ret = find_node_start(node[0], s, idxmap)

    elif isinstance(node, Module):
        ret = find_node_start(node.body[0], s, idxmap)

    elif isinstance(node, BinOp):
        leftstart = find_node_start(node.left, s, idxmap)
        if leftstart <> None:
            ret = leftstart
        else:
            ret = map_idx(idxmap, node.lineno, node.col_offset)

    elif hasattr(node, 'lineno'):
        if node.col_offset >= 0:
            ret = map_idx(idxmap, node.lineno, node.col_offset)
        else:                           # special case for """ strings
            i = map_idx(idxmap, node.lineno, node.col_offset)
            while i > 0 and i+2 < len(s) and s[i:i+3] <> '"""':
                i -= 1
            ret = i
    else:
        ret = None

    if ret == None and hasattr(node, 'lineno'):
        raise TypeError("got None for node that has lineno", node)

    if isinstance(node, AST) and ret <> None:
        node.node_start = ret

    return ret




def find_node_end(node, s, idxmap):

    if hasattr(node, 'node_end'):
        return node.node_end

    elif isinstance(node, list):
        ret = find_node_end(node[-1], s, idxmap)

    elif isinstance(node, Module):
        ret = find_node_end(node.body[-1], s, idxmap)

    elif isinstance(node, Expr):
        ret = find_node_end(node.value, s, idxmap)

    elif isinstance(node, Str):
        i = find_node_start(node, s, idxmap)
        if i+2 < len(s) and s[i:i+3] == '"""':
            q = '"""'
            i += 3
        elif s[i] == '"':
            q = '"'
            i += 1
        elif s[i] == "'":
            q = "'"
            i += 1
        else:
            print "illegal:", i, s[i]
        ret = end_seq(s, q, i)

    elif isinstance(node, Name):
        ret = find_node_start(node, s, idxmap) + len(node.id)

    elif isinstance(node, Attribute):
        ret = end_seq(s, node.attr, find_node_end(node.value, s, idxmap))

    elif isinstance(node, FunctionDef):
        # add_missing_names(node, s, idxmap)
        # ret = find_node_end(node.nameName, s, idxmap)
        ret = find_node_end(node.body, s, idxmap)

    elif isinstance(node, Lambda):
        ret = find_node_end(node.body, s, idxmap)

    elif isinstance(node, ClassDef):
        # add_missing_names(node, s, idxmap)
        # ret = find_node_end(node.nameName, s, idxmap)
        ret = find_node_end(node.body, s, idxmap)

    elif isinstance(node, Call):
        ret = match_paren(s, '(', ')', find_node_end(node.func, s, idxmap))

    elif isinstance(node, Yield):
        ret = find_node_end(node.value, s, idxmap)

    elif isinstance(node, Return):
        if node.value <> None:
            ret = find_node_end(node.value, s, idxmap)
        else:
            ret = find_node_start(node, s, idxmap) + len('return')

    elif isinstance(node, Print):
        ret = start_seq(s, '\n', find_node_start(node, s, idxmap))

    elif (isinstance(node, For) or
          isinstance(node, While) or
          isinstance(node, If) or
          isinstance(node, IfExp)):
        if node.orelse <> []:
            ret = find_node_end(node.orelse, s, idxmap)
        else:
            ret = find_node_end(node.body, s, idxmap)

    elif isinstance(node, Assign) or isinstance(node, AugAssign):
        ret = find_node_end(node.value, s, idxmap)

    elif isinstance(node, BinOp):
        ret = find_node_end(node.right, s, idxmap)

    elif isinstance(node, BoolOp):
        ret = find_node_end(node.values[-1], s, idxmap)

    elif isinstance(node, Compare):
        ret = find_node_end(node.comparators[-1], s, idxmap)

    elif isinstance(node, UnaryOp):
        ret = find_node_end(node.operand, s, idxmap)

    elif isinstance(node, Num):
        ret = find_node_start(node, s, idxmap) + len(str(node.n))

    elif isinstance(node, List):
        ret = match_paren(s, '[', ']', find_node_start(node, s, idxmap));

    elif isinstance(node, Subscript):
        ret = match_paren(s, '[', ']', find_node_start(node, s, idxmap));

    elif isinstance(node, Tuple):
        ret = find_node_end(node.elts[-1], s, idxmap)

    elif isinstance(node, Dict):
        ret = match_paren(s, '{', '}', find_node_start(node, s, idxmap));

    elif isinstance(node, TryExcept):
        if node.orelse <> []:
            ret = find_node_end(node.orelse, s, idxmap)
        elif node.handlers <> []:
            ret = find_node_end(node.handlers, s, idxmap)
        else:
            ret = find_node_end(node.body, s, idxmap)

    elif isinstance(node, ExceptHandler):
        ret = find_node_end(node.body, s, idxmap)

    elif isinstance(node, Pass):
        ret = find_node_start(node, s, idxmap) + len('pass')

    elif isinstance(node, Break):
        ret = find_node_start(node, s, idxmap) + len('break')

    elif isinstance(node, Continue):
        ret = find_node_start(node, s, idxmap) + len('continue')

    elif isinstance(node, Global):
        ret = start_seq(s, '\n', find_node_start(node, s, idxmap))

    elif isinstance(node, Import):
        ret = find_node_start(node, s, idxmap) + len('import')

    elif isinstance(node, ImportFrom):
        ret = find_node_start(node, s, idxmap) + len('from')

    else:
        # print "[find_node_end] unrecognized node:", node, "type:", type(node)
        start = find_node_start(node, s, idxmap)
        if start <> None:
            ret = start + 3
        else:
            ret = None

    if ret == None and hasattr(node, 'lineno'):
        raise TypeError("got None for node that has lineno", node)

    if isinstance(node, AST) and ret <> None:
        node.node_end = ret

    return ret




#-------------------------------------------------------------
#                    adding missing Names
#-------------------------------------------------------------

def add_missing_names(node, s, idxmap):

    if hasattr(node, 'extraAttribute'):
        return

    if isinstance(node, list):
        for n in node:
            add_missing_names(n, s, idxmap)

    elif isinstance(node, ClassDef):
        start = find_node_start(node, s, idxmap) + len('class')
        node.nameName = str2Name(s, start, idxmap)
        node._fields += ('nameName',)

    elif isinstance(node, FunctionDef):
        start = find_node_start(node, s, idxmap) + len('def')
        node.nameName = str2Name(s, start, idxmap)
        node._fields += ('nameName',)

        if node.args.vararg <> None:
            if len(node.args.args) > 0:
                vstart = find_node_end(node.args.args[-1], s, idxmap)
            else:
                vstart = find_node_end(node.nameName, s, idxmap)
            vname = str2Name(s, vstart, idxmap)
            node.varargName = vname
        else:
            node.varargName = None
        node._fields += ('varargName',)

        if node.args.kwarg <> None:
            if len(node.args.args) > 0:
                kstart = find_node_end(node.args.args[-1], s, idxmap)
            else:
                kstart = find_node_end(node.varargName, s, idxmap)
            kname = str2Name(s, kstart, idxmap)
            node.kwarg_name = kname
        else:
            node.kwarg_name = None
        node._fields += ('kwarg_name',)

    elif isinstance(node, Attribute):
        start = find_node_end(node.value, s, idxmap)
        name = str2Name(s, start, idxmap)
        node.attr_name = name
        node._fields = ('value', 'attr_name')  # remove attr for node size accuracy

    elif isinstance(node, Compare):
        node.opsName = convert_ops(node.ops, s,
                                  find_node_start(node, s, idxmap), idxmap)
        node._fields += ('opsName',)

    elif (isinstance(node, BoolOp) or
          isinstance(node, BinOp) or
          isinstance(node, UnaryOp) or
          isinstance(node, AugAssign)):
        if hasattr(node, 'left'):
            start = find_node_end(node.left, s, idxmap)
        else:
            start = find_node_start(node, s, idxmap)
        ops = convert_ops([node.op], s, start, idxmap)
        node.opName = ops[0]
        node._fields += ('opName',)

    elif isinstance(node, Import):
        nameNames = []
        next = find_node_start(node, s, idxmap) + len('import')
        name = str2Name(s, next, idxmap)
        while name <> None and next < len(s) and s[next] <> '\n':
            nameNames.append(name)
            next = name.node_end
            name = str2Name(s, next, idxmap)
        node.nameNames = nameNames
        node._fields += ('nameNames',)

    node.extraAttribute = True



#-------------------------------------------------------------
#              utilities used by improve AST functions
#-------------------------------------------------------------

# find a sequence in a string s, returning the start point
def start_seq(s, pat, start):
    try:
        return s.index(pat, start)
    except ValueError:
        return len(s)



# find a sequence in a string s, returning the end point
def end_seq(s, pat, start):
    try:
        return s.index(pat, start) + len(pat)
    except ValueError:
        return len(s)



# find matching close paren from start
def match_paren(s, open, close, start):
    while s[start] <> open and start < len(s):
        start += 1
    if start >= len(s):
        return len(s)

    left = 1
    i = start + 1
    while left > 0 and i < len(s):
        if s[i] == open:
            left += 1
        elif s[i] == close:
            left -= 1
        i += 1
    return i



# build table for lineno <-> index oonversion
def build_index_map(s):
    line = 0
    col = 0
    idx = 0
    idxmap = [0]
    while idx < len(s):
        if s[idx] == '\n':
            idxmap.append(idx + 1)
            line += 1
        idx += 1
    return idxmap



# convert (line, col) to offset index
def map_idx(idxmap, line, col):
    return idxmap[line-1] + col



# convert offset index into (line, col)
def map_line_col(idxmap, idx):
    line = 0
    for start in idxmap:
        if idx < start:
            break
        line += 1
    col = idx - idxmap[line-1]
    return (line, col)



# convert string to Name
def str2Name(s, start, idxmap):
    i = start;
    while i < len(s) and not is_alpha(s[i]):
        i += 1
    startIdx = i
    ret = []
    while i < len(s) and is_alpha(s[i]):
        ret.append(s[i])
        i += 1
    endIdx = i
    id1 = ''.join(ret)

    if id1 == '':
        return None
    else:
        name = Name(id1, None)
        name.node_start = startIdx
        name.node_end = endIdx
        name.lineno, name.col_offset = map_line_col(idxmap, startIdx)
        return name



def convert_ops(ops, s, start, idxmap):
    syms = map(lambda op: ops_map[type(op)], ops)
    i = start
    j = 0
    ret = []
    while i < len(s) and j < len(syms):
        oplen = len(syms[j])
        if s[i:i+oplen] == syms[j]:
            opName = Name(syms[j], None)
            opName.node_start = i
            opName.node_end = i+oplen
            opName.lineno, opName.col_offset = map_line_col(idxmap, i)
            ret.append(opName)
            j += 1
            i = opName.node_end
        else:
            i += 1
    return ret


# lookup table for operators for convert_ops
ops_map = {
    # compare:
    Eq     : '==',
    NotEq  : '<>',
    Lt     : '<',
    LtE    : '<=',
    Gt     : '>',
    GtE    : '>=',
    In     : 'in',
    NotIn  : 'not in',

    # BoolOp
    Or  : 'or',
    And : 'and',
    Not : 'not',

    # BinOp
    Add  : '+',
    Sub  : '-',
    Mult : '*',
    Div  : '/',
    Mod  : '%',

    # UnaryOp
    USub : '-',
    UAdd : '+',
}






#-------------------------------------------------------------
#                        HTML generation
#-------------------------------------------------------------


#-------------------- types and utilities ----------------------

class Tag:
    def __init__(self, tag, idx, start=-1):
        self.tag = tag
        self.idx = idx
        self.start = start
    def __repr__(self):
        return "tag:" + str(self.tag) + ":" + str(self.idx)



# escape for HTML
def escape(s):
    s = s.replace('"', '&quot;')
    s = s.replace("'", '&#39;')
    s = s.replace("<", '&lt;')
    s = s.replace(">", '&gt;')
    return s



uid_count = -1
uid_hash = {}
def clear_uid():
    global uid_count, uid_hash
    uid_count = -1
    uid_hash = {}


def uid(node):
    if uid_hash.has_key(node):
        return uid_hash[node]

    global uid_count
    uid_count += 1
    uid_hash[node] = str(uid_count)
    return str(uid_count)



def line_id(lineno):
    return 'L' + str(lineno);


def qs(s):
    return "'" + s + "'"



#-------------------- main HTML generating function ------------------

def gen_html(text, changes, side):
    ltags = line_tags(text)
    ctags = change_tags(text, changes, side)
    ktags = keyword_tags(side)
    body = apply_tags(text, ltags + ctags + ktags, side)

    out = []
    out.append('<html>\n')
    out.append('<head>\n')
    out.append('<META http-equiv="Content-Type" content="text/html; charset=utf-8">\n')
    out.append('<LINK href="diff.css" rel="stylesheet" type="text/css">\n')
    out.append('<script type="text/javascript" src="nav.js"></script>\n')
    out.append('</head>\n')
    out.append('<body>\n')

    out.append('<pre>\n')
    out.append(body)
    out.append('</pre>\n')

    # out.append('</body>\n')
    # out.append('</html>\n')

    return ''.join(out)



# put the tags generated by change_tags into the text and create HTML
def apply_tags(s, tags, side):
    tags = sorted(tags, key = lambda t: (t.idx, -t.start))
    curr = 0
    out = []
    for t in tags:
        while curr < t.idx and curr < len(s):
            out.append(escape(s[curr]))
            curr += 1
        out.append(t.tag)

    while curr < len(s):
        out.append(escape(s[curr]))
        curr += 1
    return ''.join(out)




#--------------------- tag generation functions ----------------------

def change_tags(s, changes, side):
    tags = []
    for r in changes:
        key = r.orig if side == 'left' else r.cur
        if hasattr(key, 'lineno'):
            start = node_start(key)
            if isinstance(key, FunctionDef):
                end = start + len('def')
            elif isinstance(key, ClassDef):
                end = start + len('class')
            else:
                end = node_end(key)

            if r.orig <> None and r.cur <> None:
                # <a ...> for change and move
                tags.append(Tag(link_tag_start(r, side), start))
                tags.append(Tag("</a>", end, start))
            else:
                # <span ...> for deletion and insertion
                tags.append(Tag(span_start(r), start))
                tags.append(Tag('</span>', end, start))

    return tags



def line_tags(s):
    out = []
    lineno = 1;
    curr = 0
    while curr < len(s):
        if curr == 0 or s[curr-1] == '\n':
            out.append(Tag('<div class="line" id="L' + str(lineno) + '">', curr))
            out.append(Tag('<span class="lineno">' + str(lineno) + ' </span>', curr))
        if s[curr] == '\n':
            out.append(Tag('</div>', curr))
            lineno += 1
        curr += 1
    out.append(Tag('</div>', curr))
    return out



def keyword_tags(side):
    tags = []
    allNodes = allNodes1 if side == 'left' else allNodes2
    for node in allNodes:
        if type(node) in kwd_map:
            kw = kwd_map[type(node)]
            start = node_start(node)
            if src(node)[:len(kw)] == kw:
                startTag = (Tag('<span class="keyword">', start))
                tags.append(startTag)
                endTag = Tag('</span>', start + len(kw), start)
                tags.append(endTag)
    return tags


def span_start(diff):
    if diff.cur == None:
        cls = "deletion"
    else:
        cls = "insertion"
    text = escape(describe_change(diff))
    return '<span class="' + cls + '" title="' + text + '">'



def link_tag_start(diff, side):
    if side == 'left':
        me, other = diff.orig, diff.cur
    else:
        me, other = diff.cur, diff.orig

    text = escape(describe_change(diff))
    if diff.cost > 0:
        cls = "change"
    else:
        cls = "move"

    return ('<a id="' + uid(me) + '" '
            + ' class="' + cls + '" '
            + ' title="' + text + '" '
            + 'onclick="highlight('
                          + qs(uid(me)) + ","
                          + qs(uid(other)) + ","
                          + qs(line_id(me.lineno)) + ","
                          + qs(line_id(other.lineno)) + ')">')


kwd_map = {
    FunctionDef : 'def',
    ClassDef    : 'class',
    For         : 'for',
    While       : 'while',
    If          : 'if',
    With        : 'with',
    Return      : 'return',
    Yield       : 'yield',
    Global      : 'global',
    Raise       : 'raise',
    Pass        : 'pass',
    TryExcept   : 'try',
    TryFinally  : 'try',
    }




# human readable description of node

def describe_node(node):

    def code(s):
        return "'" + s + "'"

    def short(node):
        if isinstance(node, Module):
            ret = "module"
        elif isinstance(node, Import):
            ret = "import statement"
        elif isinstance(node, Name):
            ret = code(node.id)
        elif isinstance(node, Attribute):
            ret = code(short(node.value) + "." + short(node.attr_name))
        elif isinstance(node, FunctionDef):
            ret = "function " + code(node.name)
        elif isinstance(node, ClassDef):
            ret = "class " + code(node.name)
        elif isinstance(node, Call):
            ret = "call to " + code(short(node.func))
        elif isinstance(node, Assign):
            ret = "assignment"
        elif isinstance(node, If):
            ret = "if statement"
        elif isinstance(node, While):
            ret = "while loop"
        elif isinstance(node, For):
            ret = "for loop"
        elif isinstance(node, Yield):
            ret = "yield"
        elif isinstance(node, TryExcept) or isinstance(node, TryFinally):
            ret = "try statement"
        elif isinstance(node, Compare):
            ret = "comparison " + src(node)
        elif isinstance(node, Return):
            ret = "return " + short(node.value)
        elif isinstance(node, Print):
            ret = ("print " + short(node.dest) +
                   ", " if (node.dest!=None) else "" + print_list(node.values))
        elif isinstance(node, Expr):
            ret = "expression " + short(node.value)
        elif isinstance(node, Num):
            ret = str(node.n)
        elif isinstance(node, Str):
            if len(node.s) > 20:
                ret = "string " + code(node.s[:20]) + "..."
            else:
                ret = "string " + code(node.s)
        elif isinstance(node, Tuple):
            ret = "tuple (" + src(node) + ")"
        elif isinstance(node, BinOp):
            ret = (short(node.left) + " " +
                   node.opName.id + " " + short(node.right))
        elif isinstance(node, BoolOp):
            ret = src(node)
        elif isinstance(node, UnaryOp):
            ret = node.opName.id + " " + short(node.operand)
        elif isinstance(node, Pass):
            ret = "pass"
        elif isinstance(node, list):
            ret = map(short, node)
        else:
            ret = str(type(node))
        return ret

    ret = short(node)
    if hasattr(node, 'lineno'):
        ret = re.sub(" *(line [0-9]+)", '', ret)
        return ret + " (line " + str(node.lineno) + ")"
    else:
        return ret




# describe a change in a human readable fashion
def describe_change(diff):

    ratio = diff.similarity()
    sim = str(ratio)

    if ratio == 1.0:
        sim = " (unchanged)"
    else:
        sim = " (similarity %.1f%%)" % (ratio * 100)

    if diff.is_frame:
        wrap = "wrap "
    else:
        wrap = ""

    if diff.cur == None:
        ret = wrap + describe_node(diff.orig) + " deleted"
    elif diff.orig == None:
        ret = wrap + describe_node(diff.cur) + " inserted"
    elif node_name(diff.orig) <> node_name(diff.cur):
        ret = (describe_node(diff.orig) +
               " renamed to " + describe_node(diff.cur) + sim)
    elif diff.cost == 0 and diff.orig.lineno <> diff.cur.lineno:
        ret = (describe_node(diff.orig) +
               " moved to " + describe_node(diff.cur) + sim)
    elif diff.cost == 0:
        ret = describe_node(diff.orig) + " unchanged"
    else:
        ret = (describe_node(diff.orig) +
               " changed to " + describe_node(diff.cur) + sim)

    return ret





#-------------------------------------------------------------
#                     main HTML based command
#-------------------------------------------------------------

def diff(file1, file2, move=True):

    import time
    print("\nJob started at %s, %s\n" % (time.ctime(), time.tzname[0]))
    start_time = time.time()
    checkpoint(start_time)

    cleanup()

    # base files names
    base1 = base_name(file1)
    base2 = base_name(file2)

    # get AST of file1
    f1 = open(file1, 'r');
    lines1 = f1.read()
    f1.close()
    node1 = parse(lines1)
    improve_ast(node1, lines1, file1, 'left')

    # get AST of file2
    f2 = open(file2, 'r');
    lines2 = f2.read()
    f2.close()
    node2 = parse(lines2)
    improve_ast(node2, lines2, file2, 'right')


    print("[parse] finished in %s. Now start to diff." % sec2min(checkpoint()))

    # get the changes

    (changes, cost) = diff_node(node1, node2, nil, nil, 0, False)

    print ("\n[diff] processed %d nodes in %s."
           % (stat.diff_count, sec2min(checkpoint())))

    if move:
#        print "changes:", changes
        (changes, cost) = closure((changes, cost))

        print("\n[closure] finished in %s." % sec2min(checkpoint()))



    #---------------------- print final stats ---------------------
    size1 = node_size(node1)
    size2 = node_size(node2)
    total = size1 + size2

    report = ""
    report += ("\n--------------------- summary -----------------------") + "\n"
    report += ("- total changes (chars):  %d" % cost)                  + "\n"
    report += ("- total code size:        %d (left: %d  right: %d)"
               % (total, size1, size2))                                + "\n"
    report += ("- total moved pieces:     %d" % stat.move_count)        + "\n"
    report += ("- percentage of change:   %.1f%%"
               % (div(cost, total) * 100))                             + "\n"
    report += ("-----------------------------------------------------")   + "\n"

    print report


    #---------------------- generation HTML ---------------------
    # write left file
    left_changes = filterlist(lambda p: p.orig <> None, changes)
    html1 = gen_html(lines1, left_changes, 'left')

    outname1 = base1 + '.html'
    outfile1 = open(outname1, 'w')
    outfile1.write(html1)
    outfile1.write('<div class="stats"><pre class="stats">')
    outfile1.write(report)
    outfile1.write('</pre></div>')
    outfile1.write('</body>\n')
    outfile1.write('</html>\n')
    outfile1.close()


    # write right file
    right_changes = filterlist(lambda p: p.cur <> None, changes)
    html2 = gen_html(lines2, right_changes, 'right')

    outname2 = base2 + '.html'
    outfile2 = open(outname2, 'w')
    outfile2.write(html2)
    outfile2.write('<div class="stats"><pre class="stats">')
    outfile2.write(report)
    outfile2.write('</pre></div>')
    outfile2.write('</body>\n')
    outfile2.write('</html>\n')
    outfile2.close()


    # write frame file
    framename = base1 + "-" + base2 + ".html"
    framefile = open(framename, 'w')
    framefile.write('<frameset cols="50%,50%">\n')
    framefile.write('<frame name="left" src="' + base1 + '.html">\n')
    framefile.write('<frame name="right" src="' + base2 + '.html">\n')
    framefile.write('</frameset>\n')
    framefile.close()

    dur = time.time() - start_time
    print("\n[summary] Job finished at %s, %s" %
          (time.ctime(), time.tzname[0]))
    print("\n\tTotal duration: %s" % sec2min(dur))




def cleanup():
    clear_str_dist_cache()
    clear_uid()

    global allNodes1, allNodes2
    allNodes1 = set()
    allNodes2 = set()

    stat.diff_count = 0
    stat.move_count = 0
    stat.move_savings = 0



def sec2min(s):
    if s < 60:
        return ("%.1f seconds" % s)
    else:
        return ("%.1f minutes" % div(s, 60))



last_checkpoint = None
def checkpoint(init=None):
    import time
    global last_checkpoint
    if init <> None:
        last_checkpoint = init
        return None
    else:
        dur = time.time() - last_checkpoint
        last_checkpoint = time.time()
        return dur




#-------------------------------------------------------------
#                      text-based interfaces
#-------------------------------------------------------------

## text-based main command
def print_diff(file1, file2):
    (m, c) = diff_file(file1, file2)
    print "----------", file1, "<<<", c, ">>>", file2, "-----------"

    ms = pylist(m)
    ms = sorted(ms, key=lambda d: node_start(d.orig))
    print "\n-------------------- changes(", len(ms), ")---------------------- "
    for m0 in ms:
        print m0

    print "\n-------------------  end  ----------------------- "




def diff_file(file1, file2):
    node1 = parse_file(file1)
    node2 = parse_file(file2)
    return closure(diff_node(node1, node2, nil, nil, 0, False))




# printing support for debugging use
def iter_fields(node):
    """Iterate over all existing fields, excluding 'ctx'."""
    for field in node._fields:
        try:
            if field <> 'ctx':
                yield field, getattr(node, field)
        except AttributeError:
            pass


def dump(node, annotate_fields=True, include_attributes=False):
    def _format(node):
        if isinstance(node, AST):
            fields = [(a, _format(b)) for a, b in iter_fields(node)]
            rv = '%s(%s' % (node.__class__.__name__, ', '.join(
                ('%s=%s' % field for field in fields)
                if annotate_fields else
                (b for a, b in fields)
            ))
            if include_attributes and node._attributes:
                rv += fields and ', ' or ' '
                rv += ', '.join('%s=%s' % (a, _format(getattr(node, a)))
                                for a in node._attributes)
            return rv + ')'
        elif isinstance(node, list):
            return '[%s]' % ', '.join(_format(x) for x in node)
        return repr(node)
    if not isinstance(node, AST):
        raise TypeError('expected AST, got %r' % node.__class__.__name__)
    return _format(node)

def print_list(ls):
    if (ls == None or ls == []):
        return ""
    elif (len(ls) == 1):
        return str(ls[0])
    else:
        return str(ls)




# for debugging use
def printAst(node):
    if (isinstance(node, Module)):
        ret = "module:" + str(node.body)
    elif (isinstance(node, Name)):
        ret = str(node.id)
    elif (isinstance(node, Attribute)):
        if hasattr(node, 'attr_name'):
            ret = str(node.value) + "." + str(node.attr_name)
        else:
            ret = str(node.value) + "." + str(node.attr)
    elif (isinstance(node, FunctionDef)):
        if hasattr(node, 'nameName'):
            ret = "fun:" + str(node.nameName)
        else:
            ret = "fun:" + str(node.name)
    elif (isinstance(node, ClassDef)):
        ret = "class:" + str(node.name)
    elif (isinstance(node, Call)):
        ret = "call:" + str(node.func) + ":(" + print_list(node.args) + ")"
    elif (isinstance(node, Assign)):
        ret = "(" + print_list(node.targets) + " <- " + printAst(node.value) + ")"
    elif (isinstance(node, If)):
        ret = "if " + str(node.test) + ":" + print_list(node.body) + ":" + print_list(node.orelse)
    elif (isinstance(node, Compare)):
        ret = str(node.left) + ":" + print_list(node.ops) + ":" + print_list(node.comparators)
    elif (isinstance(node, Return)):
        ret = "return " + repr(node.value)
    elif (isinstance(node, Print)):
        ret = "print(" + (str(node.dest) + ", " if (node.dest!=None) else "") + print_list(node.values) + ")"
    elif (isinstance(node, Expr)):
        ret = "expr:" + str(node.value)
    elif (isinstance(node, Num)):
        ret = "num:" + str(node.n)
    elif (isinstance(node, Str)):
        ret = 'str:"' + str(node.s) + '"'
    elif (isinstance(node, BinOp)):
        ret = str(node.left) + " " + str(node.op) + " " + str(node.right)
    elif (isinstance(node, Add)):
        ret = '+'
    elif (isinstance(node, Mult)):
        ret = '*'
    elif isinstance(node, NotEq):
        ret = '<>'
    elif (isinstance(node, Eq)):
        ret = '=='
    elif (isinstance(node, Pass)):
        ret = "pass"
    elif isinstance(node,list):
        ret = print_list(node)
    else:
        ret = str(type(node))

    if hasattr(node, 'lineno'):
        return re.sub("@[0-9]+", '', ret) + "@" + str(node.lineno)
    elif hasattr(node, 'node_start'):
        return re.sub("@[0-9]+", '', ret) + "%" + str(node_start(node))
    else:
        return ret


def install_printer():
    import inspect, ast
    for name, obj in inspect.getmembers(ast):
        if (inspect.isclass(obj) and not (obj == AST)):
            obj.__repr__ = printAst

install_printer()


## if run under command line
## pydiff.py file1.py file2.py
if len(sys.argv) == 3:
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    diff(file1, file2)
