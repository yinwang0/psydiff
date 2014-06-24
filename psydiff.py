#!/usr/bin/env python

import sys
import time
import cProfile

from ast import *

from parameters import *
from improve_ast import *
from htmlize import *
from utils import *



#------------------------------- types ------------------------------
class Stat:
    "storage for stat counters"
    def __init__(self):
        self.reset()

    def reset(self):
        self.diff_count = 0
        self.move_count = 0
        self.move_savings = 0

    def add_moves(self, nterms):
        self.move_savings += nterms
        self.move_count +=1
        if self.move_count % 1000 == 0:
            dot()
    def add_diff(self):
        self.diff_count += 1
        if stat.diff_count % 1000 == 0:
            dot()

stat = Stat()



# The difference between nodes are stored as a Change structure.
class Change:
    def __init__(self, orig, cur, cost, is_frame=False):
        self.orig = orig
        self.cur = cur
        if orig is None:
            self.cost = node_size(cur)
        elif cur is None:
            self.cost = node_size(orig)
        elif cost == 'all':
            self.cost = node_size(orig) + node_size(cur)
        else:
            self.cost = cost
        self.is_frame = is_frame
    def __repr__(self):
        fr = "F" if self.is_frame else "-"
        def hole(x):
            return [] if x==None else x
        return ("(C:" + str(hole(self.orig)) + ":" + str(hole(self.cur))
                + ":" + str(self.cost) + ":" + str(self.similarity())
                + ":" + fr + ")")
    def similarity(self):
        total = node_size(self.orig) + node_size(self.cur)
        return 1 - div(self.cost, total)



# Three major kinds of changes:
# * modification
# * deletion
# * insertion
def mod_node(node1, node2, cost):
    return Change(node1, node2, cost)

def del_node(node):
    return Change(node, None, node_size(node))

def ins_node(node):
    return Change(None, node, node_size(node))


# 2-D array table for memoization of dynamic programming
def create_table(x, y):
    table = []
    for i in range(x+1):
        table.append([None] * (y+1))
    return table

def table_lookup(t, x, y):
    return t[x][y]

def table_put(t, x, y, v):
    t[x][y] = v





#-------------------------------------------------------------
#                  string distance function
#-------------------------------------------------------------

### diff cache for AST nodes
str_dist_cache = {}


### string distance function
def str_dist(s1, s2):
    cached = str_dist_cache.get((s1, s2))
    if cached is not None:
        return cached

    if len(s1) > 100 or len(s2) > 100:
        if s1 != s2:
            return 2.0
        else:
            return 0

    table = create_table(len(s1), len(s2))
    d = dist1(table, s1, s2)
    ret = div(2*d, len(s1) + len(s2))

    str_dist_cache[(s1, s2)]=ret
    return ret


# the main dynamic programming part
# similar to the structure of diff_list
def dist1(table, s1, s2):
    def memo(v):
        table_put(table, len(s1), len(s2), v)
        return v

    cached = table_lookup(table, len(s1), len(s2))
    if cached is not None:
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

def diff_node(node1, node2, depth, move):

    # try substructural diff
    def trysub(cc):
        (changes, cost) = cc
        if not move:
            return (changes, cost)
        elif can_move(node1, node2, cost):
            return (changes, cost)
        else:
            mc1 = diff_subnode(node1, node2, depth, move)
            if mc1 is not None:
                return mc1
            else:
                return (changes, cost)

    if isinstance(node1, list) and not isinstance(node2, list):
        node2 = [node2]

    if not isinstance(node1, list) and isinstance(node2, list):
        node1 = [node1]

    if isinstance(node1, list) and isinstance(node2, list):
        node1 = serialize_if(node1)
        node2 = serialize_if(node2)
        table = create_table(len(node1), len(node2))
        return diff_list(table, node1, node2, 0, move)

    # statistics
    stat.add_diff()

    if node1 == node2:
        return ([mod_node(node1, node2, 0)], 0)

    if isinstance(node1, Num) and isinstance(node2, Num):
        if node1.n == node2.n:
            return ([mod_node(node1, node2, 0)], 0)
        else:
            return ([mod_node(node1, node2, 1)], 1)

    if isinstance(node1, Str) and isinstance(node2, Str):
        cost = str_dist(node1.s, node2.s)
        return ([mod_node(node1, node2, cost)], cost)

    if (isinstance(node1, Name) and isinstance(node2, Name)):
        cost = str_dist(node1.id, node2.id)
        return ([mod_node(node1, node2, cost)], cost)

    if (isinstance(node1, Attribute) and isinstance(node2, Name) or
        isinstance(node1, Name) and isinstance(node2, Attribute) or
        isinstance(node1, Attribute) and isinstance(node2, Attribute)):
        s1 = attr_to_str(node1)
        s2 = attr_to_str(node2)
        if s1 is not None and s2 is not None:
            cost = str_dist(s1, s2)
            return ([mod_node(node1, node2, cost)], cost)
        # else fall through for things like f(x).y vs x.y

    if isinstance(node1, Module) and isinstance(node2, Module):
        return diff_node(node1.body, node2.body, depth, move)

    # same type of other AST nodes
    if (isinstance(node1, AST) and isinstance(node2, AST) and
        type(node1) == type(node2)):

        fs1 = node_fields(node1)
        fs2 = node_fields(node2)
        changes, cost = [], 0
        min_len = min(len(fs1), len(fs2))

        for i in range(min_len):
            (m, c) = diff_node(fs1[i], fs2[i], depth, move)
            changes = m + changes
            cost += c

        # final all moves local to the node
        return find_moves((changes, cost))

    if (type(node1) == type(node2) and
             is_empty_container(node1) and is_empty_container(node2)):
        return ([mod_node(node1, node2, 0)], 0)

    # all unmatched types and unequal values
    return trysub(([del_node(node1), ins_node(node2)],
                   node_size(node1) + node_size(node2)))



########################## diff of a list ##########################

# diff_list is the main part of dynamic programming

def diff_list(table, ls1, ls2, depth, move):

    def memo(v):
        table_put(table, len(ls1), len(ls2), v)
        return v

    def guess(table, ls1, ls2):
        (m0, c0) = diff_node(ls1[0], ls2[0], depth, move)
        (m1, c1) = diff_list(table, ls1[1:], ls2[1:], depth, move)
        cost1 = c1 + c0

        if ((is_frame(ls1[0]) and
             is_frame(ls2[0]) and
             not node_framed(ls1[0], m0) and
             not node_framed(ls2[0], m0))):
            frame_change = [mod_node(ls1[0], ls2[0], c0)]
        else:
            frame_change = []

        # short cut 1 (func and classes with same names)
        if can_move(ls1[0], ls2[0], c0):
            return (frame_change + m0 + m1, cost1)

        else:  # do more work
            (m2, c2) = diff_list(table, ls1[1:], ls2, depth, move)
            (m3, c3) = diff_list(table, ls1, ls2[1:], depth, move)
            cost2 = c2 + node_size(ls1[0])
            cost3 = c3 + node_size(ls2[0])

            if (not different_def(ls1[0], ls2[0]) and
                cost1 <= cost2 and cost1 <= cost3):
                return (frame_change + m0 + m1, cost1)
            elif (cost2 <= cost3):
                return ([del_node(ls1[0])] + m2, cost2)
            else:
                return ([ins_node(ls2[0])] + m3, cost3)

    # cache look up
    cached = table_lookup(table, len(ls1), len(ls2))
    if cached is not None:
        return cached

    if (ls1 == [] and ls2 == []):
        return memo(([], 0))

    elif (ls1 != [] and ls2 != []):
        return memo(guess(table, ls1, ls2))

    elif ls1 == []:
        d = []
        for n in ls2:
            d = [ins_node(n)] + d
        return memo((d, node_size(ls2)))

    else: # ls2 == []:
        d = []
        for n in ls1:
            d = [del_node(n)] + d
        return memo((d, node_size(ls1)))




###################### diff into a subnode #######################

# Subnode diff is only used in the moving phase. There is no
# need to compare the substructure of two nodes in the first
# run, because they will be reconsidered if we just consider
# them to be complete deletion and insertions.

def diff_subnode(node1, node2, depth, move):

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
                (m0, c0) = diff_node(node1, f, depth+1, move)
                if can_move(node1, f, c0):
                    if not isinstance(f, list):
                        m1 = [mod_node(node1, f, c0)]
                    else:
                        m1 = []
                    framecost = node_size(node2) - node_size(node1)
                    m2 = [Change(None, node2, framecost, True)]
                    return (m2 + m1 + m0, c0 + framecost)

        if (node_size(node1) > node_size(node2)):
            for f in node_fields(node1):
                (m0, c0) = diff_node(f, node2, depth+1, move)
                if can_move(f, node2, c0):
                    framecost = node_size(node1) - node_size(node2)
                    if not isinstance(f, list):
                        m1 = [mod_node(f, node2, c0)]
                    else:
                        m1 = []
                    m2 = [Change(node1, None, framecost, True)]
                    return (m2 + m1 + m0, c0 + framecost)

    return None




##########################################################################
##                          move detection
##########################################################################
def move_candidate(node):
    return (is_def(node) or node_size(node) >= MOVE_SIZE)


def match_up(changes, round=0):

    deletions = lfilter(lambda p: (p.cur is None and
                                  move_candidate(p.orig) and
                                  not p.is_frame),
                       changes)

    insertions = lfilter(lambda p: (p.orig is None and
                                   move_candidate(p.cur) and
                                   not p.is_frame),
                        changes)

    matched = []
    new_changes = []
    total = 0

    # find definition with the same names first
    for d0 in deletions:
        for a0 in insertions:
            (node1, node2) = (d0.orig, a0.cur)
            if same_def(node1, node2):
                matched.append(d0)
                matched.append(a0)
                deletions.remove(d0)
                insertions.remove(a0)

                (changes, cost) = diff_node(node1, node2, 0, True)
                nterms = node_size(node1) + node_size(node2)
                new_changes.extend(changes)
                total += cost

                if (not node_framed(node1, changes) and
                    not node_framed(node2, changes) and
                    is_def(node1) and is_def(node2)):
                    new_changes.append(mod_node(node1, node2, cost))
                stat.add_moves(nterms)
                break


    # match the rest of the deltas
    for d0 in deletions:
        for a0 in insertions:
            (node1, node2) = (d0.orig, a0.cur)
            (changes, cost) = diff_node(node1, node2, 0, True)
            nterms = node_size(node1) + node_size(node2)

            if (cost <= (node_size(node1) + node_size(node2)) * MOVE_RATIO or
                node_framed(node1, changes) or
                node_framed(node2, changes)):

                matched.append(d0)
                matched.append(a0)
                insertions.remove(a0)
                new_changes.extend(changes)
                total += cost

                if (not node_framed(node1, changes) and
                    not node_framed(node2, changes) and
                    is_def(node1) and is_def(node2)):
                    new_changes.append(mod_node(node1, node2, cost))
                stat.add_moves(nterms)
                break

    return (matched, new_changes, total)



# Get moves repeatedly because new moves may introduce new
# deletions and insertions.

def find_moves(res):
    (changes, cost) = res
    matched = None
    move_round = 1

    while move_round <= MOVE_ROUND and matched != []:
        (matched, new_changes, c) = match_up(changes, move_round)
        move_round += 1
        changes = lfilter(lambda c: c not in matched, changes)
        changes.extend(new_changes)
        savings = sum(map(lambda p: node_size(p.orig) + node_size(p.cur), matched))
        cost = cost + c - savings
    return (changes, cost)




#-------------------------------------------------------------
#                     main diff command
#-------------------------------------------------------------

def diff(file1, file2, move=True):

    print("File 1: %s" % file1)
    print("File 2: %s" % file2)
    print("Start time: %s, %s" % (time.ctime(), time.tzname[0]))
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

    try:
        node1 = parse(lines1)
    except Exception:
        print('file %s cannot be parsed' % file1)
        exit(-1)

    improve_ast(node1, lines1, file1, 'left')

    # get AST of file2
    f2 = open(file2, 'r');
    lines2 = f2.read()
    f2.close()

    try:
        node2 = parse(lines2)
    except Exception:
        print('file %s cannot be parsed' % file2)
        exit(-1)

    improve_ast(node2, lines2, file2, 'right')

    print("Parse finished in %s. Now start to diff." % sec_to_min(checkpoint()))

    # get the changes

    (changes, cost) = diff_node(node1, node2, 0, False)

    print("\n[diff] processed %d nodes in %s."
          % (stat.diff_count, sec_to_min(checkpoint())))


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

    print(report)


    #---------------------- generation HTML ---------------------

    htmlize(changes, file1, file2, lines1, lines2)

    dur = time.time() - start_time
    print("\n[summary] Job finished at %s, %s" %
          (time.ctime(), time.tzname[0]))
    print("\n\tTotal duration: %s" % sec_to_min(dur))




def cleanup():
    str_dist_cache.clear()
    clear_uid()

    global allNodes1, allNodes2
    allNodes1 = set()
    allNodes2 = set()

    stat.reset()



def sec_to_min(s):
    if s < 60:
        return ("%.1f seconds" % s)
    else:
        return ("%.1f minutes" % div(s, 60))



last_checkpoint = None
def checkpoint(init=None):
    import time
    global last_checkpoint
    if init is not None:
        last_checkpoint = init
        return None
    else:
        dur = time.time() - last_checkpoint
        last_checkpoint = time.time()
        return dur




#-------------------------------------------------------------
#                      text-based interfaces
#-------------------------------------------------------------

## print the diffs as text
def print_diff(file1, file2):
    (m, c) = diff_file(file1, file2)
    print("----------", file1, "<<<", c, ">>>", file2, "-----------")

    ms = m
    ms = sorted(ms, key=lambda d: node_start(d.orig))
    print("\n-------------------- changes(", len(ms), ")---------------------- ")
    for m0 in ms:
        print(m0)

    print("\n-------------------  end  ----------------------- ")




def diff_file(file1, file2):
    node1 = parse_file(file1)
    node2 = parse_file(file2)
    return diff_node(node1, node2, 0)


def main():
    ## if run under command line
    ## psydiff.py file1.py file2.py
    if len(sys.argv) == 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        diff(file1, file2)


if __name__ == '__main__':
    main()
