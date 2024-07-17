import ast
import inspect
from sympy import symbols, expand, factor, simplify, sympify, Add, SympifyError

class ExpressionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.operations = []

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            self.operations.append(('add', self.get_node_representation(node.left), self.get_node_representation(node.right)))
        elif isinstance(node.op, ast.Sub):
            self.operations.append(('sub', self.get_node_representation(node.left), self.get_node_representation(node.right)))
        elif isinstance(node.op, ast.Mult):
            self.operations.append(('mult', self.get_node_representation(node.left), self.get_node_representation(node.right)))
        elif isinstance(node.op, ast.Div):
            self.operations.append(('div', self.get_node_representation(node.left), self.get_node_representation(node.right)))
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        op = self.get_op_symbol(node.op)
        operand = self.get_node_representation(node.operand)
        self.operations.append(('unary', op, operand))
        self.generic_visit(node)

    def visit_Name(self, node):
        self.operations.append(('name', node.id))

    def visit_Constant(self, node):
        self.operations.append(('const', node.value))

    def visit_Call(self, node):
        self.operations.append(('call', node.func.id))
        self.generic_visit(node)
    
    def visit_Subscript(self, node):
        value = self.get_node_representation(node.value)
        index = self.get_node_representation(node.slice)
        self.operations.append(('subscript', value, index))
        self.generic_visit(node)
    
    def get_node_representation(self, node):
        if isinstance(node, ast.Name):
            return ('name', node.id)
        elif isinstance(node, ast.Constant):
            return ('const', node.value)
        elif isinstance(node, ast.Subscript):
            value = self.get_node_representation(node.value)
            index = self.get_node_representation(node.slice)
            return ('subscript', value, index)
        elif isinstance(node, ast.BinOp):
            left = self.get_node_representation(node.left)
            right = self.get_node_representation(node.right)
            op = self.get_op_symbol(node.op)
            return (op, left, right)
        elif isinstance(node, ast.UnaryOp):
            op = self.get_op_symbol(node.op)
            operand = self.get_node_representation(node.operand)
            return ('unary', op, operand)
        elif isinstance(node, ast.Call):
            func = node.func.id
            args = [self.get_node_representation(arg) for arg in node.args]
            return ('call', func, args)
        return str(node)
    
    def get_op_symbol(self, op):
        if isinstance(op, ast.Add):
            return 'add'
        elif isinstance(op, ast.Sub):
            return 'sub'
        elif isinstance(op, ast.Mult):
            return 'mult'
        elif isinstance(op, ast.Div):
            return 'div'
        elif isinstance(op, ast.UAdd):
            return '+'
        elif isinstance(op, ast.USub):
            return '-'
        elif isinstance(op, ast.Invert):
            return '~'
        return '?'

def analyze_expression(expression):
    tree = ast.parse(expression, mode='eval')
    visitor = ExpressionVisitor()
    visitor.visit(tree)
    return visitor.operations

def extract_expressions_from_function(func):
    source_code = inspect.getsource(func)
    tree = ast.parse(source_code)
    expressions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            if isinstance(value, ast.Call):
                continue
            expr = ast.unparse(value)
            expressions.append(expr)
    
    return expressions

def generate_minimal_representation(expressions):
    all_operations = [analyze_expression(expr) for expr in expressions]
    
    unique_operations = set()
    for ops in all_operations:
        unique_operations.update(ops)
    
    minimal_representation = list(unique_operations)
    
    return minimal_representation, all_operations

def generate_minimal_expression(expressions):
    sympy_expressions = []

    for expr in expressions:
        sympy_expr_str = expr_to_sympy(expr)
        try:
            sympy_expr = sympify(sympy_expr_str, evaluate=False)
            sympy_expressions.append(sympy_expr)
        except SympifyError as e:
            print(f"SympifyError: Could not parse {sympy_expr_str} - {e}")

    common_expr = Add(*sympy_expressions)
    common_expr = expand(common_expr)
    common_expr = factor(common_expr)
    common_expr = simplify(common_expr)
    
    return common_expr

def expr_to_sympy(expr):
    if isinstance(expr, list) and len(expr) == 1:
        expr = expr[0]

    expr_str = ""
    if expr[0] == 'name':
        expr_str += expr[1]
    elif expr[0] == 'const':
        expr_str += str(expr[1])
    elif expr[0] in ['add', 'sub', 'mult', 'div']:
        left = expr_to_sympy([expr[1]])
        right = expr_to_sympy([expr[2]])
        op = {'add': '+', 'sub': '-', 'mult': '*', 'div': '/'}[expr[0]]
        expr_str += f"({left} {op} {right})"
    elif expr[0] == 'unary':
        expr_str += f"({expr[1]}{expr_to_sympy([expr[2]])})"
    elif expr[0] == 'subscript':
        value = expr_to_sympy([expr[1]])
        index = expr_to_sympy([expr[2]])
        expr_str += f"{value}[{index}]"
    elif expr[0] == 'call':
        args = ', '.join([expr_to_sympy([arg]) for arg in expr[2]])
        expr_str += f"{expr[1]}({args})"
    
    return expr_str

def dxdtfunc(outarray, state, constants, t):
    ref = clamp(cos(constants[13] * t) * constants[9], constants[10])
    control = linear_control_eq(constants[11], constants[12], state[4], constants[10])

    outarray[0] = state[1]
    outarray[1] = (-state[0] - constants[3] * state[1] + constants[0] * state[2] + constants[7] * ref)
    outarray[2] = (-constants[1] * state[2] + constants[2] * state[3] * state[3])
    outarray[3] = (-constants[6] * state[3] + constants[6] * control)
    outarray[4] = (-constants[5] * state[4] + constants[8] * state[1])

expressions = extract_expressions_from_function(dxdtfunc)
_, all_operations = generate_minimal_representation(expressions)

minimal_expression = generate_minimal_expression(all_operations)

print(f"Minimal Expression: {minimal_expression}\n")
