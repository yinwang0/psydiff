#-------------------------------------------------------------
#                   improvements to the AST
#-------------------------------------------------------------

from ast import *
from utils import *
from parameters import *


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

