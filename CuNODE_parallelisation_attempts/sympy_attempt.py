import ast
import inspect
import re
from sympy import symbols, simplify, sympify, cse, Symbol
import sympy
from sympy.simplify.cse_main import opt_cse

# Function to extract expressions as strings
def extract_expressions_from_function(func):
    source_code = inspect.getsource(func)
    tree = ast.parse(source_code)
    expressions = []

    class ExpressionExtractor(ast.NodeVisitor):
        def visit_Assign(self, node):
            expr = ast.unparse(node.value)
            expressions.append(expr)
            self.generic_visit(node)

    extractor = ExpressionExtractor()
    extractor.visit(tree)
    return expressions

# Function to replace subscript notations
def replace_subscripts(expr_str):
    # Use a regex to find all instances of `name[index]`
    pattern = re.compile(r'(\w+)\[(\d+)\]')
    replaced_expr = pattern.sub(r'\1_\2', expr_str)
    return replaced_expr

# Function to simplify expressions
def simplify_expressions(expressions):
    sympy_expressions = []
    for expr in expressions:
        try:
            replaced_expr = replace_subscripts(expr)
            sympy_expr = sympify(replaced_expr)
            sympy_expressions.append(sympy_expr)
        except SympifyError as e:
            print(f"SympifyError: Could not parse {expr} - {e}")
    return sympy_expressions

# Example function
def dxdtfunc(outarray, state, constants, t):
    # ref = clamp(cos(constants[13] * t) * constants[9], constants[10])
    # control = linear_control_eq(constants[11], constants[12], state[4], constants[10])

    outarray[0] = state[1]
    outarray[1] = (-state[0] - constants[3] * state[1] + constants[0] * state[2] + constants[7] * ref)
    outarray[2] = (-constants[1] * state[2] + constants[2] * state[3] * state[3])
    outarray[3] = (-constants[6] * state[3] + constants[6] * control)
    outarray[4] = (-constants[5] * state[4] + constants[8] * state[1])

# Extract and simplify expressions
expressions = extract_expressions_from_function(dxdtfunc)
sympy_expressions = simplify_expressions(expressions)

for expr in sympy_expressions:
    print(sympy.simplify(expr))
    print(sympy.separatevars(expr))
    print(sympy.expand(expr))

test =     opt_cse(sympy_expressions)
print(opt_cse(sympy_expressions))
print(opt_cse([expr]))
