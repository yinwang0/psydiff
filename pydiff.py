#!/usr/bin/env python

import sys
import re
import cProfile

from ast import *
from lists import *

from improve_ast import *
from htmlize import *
from utils import *
from parameters import *



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
