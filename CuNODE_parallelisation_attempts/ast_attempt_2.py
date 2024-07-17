import ast
import inspect
from sympy import symbols, expand, factor, simplify, sympify, Add, SympifyError, Symbol
from sympy.core.operations import AssocOp

class ExpressionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.operations = []

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self.get_op_symbol(node.op)
        self.operations.append((op, left, right))
        return (op, left, right)

    def visit_UnaryOp(self, node):
        op = self.get_op_symbol(node.op)
        operand = self.visit(node.operand)
        self.operations.append(('unary', op, operand))
        return ('unary', op, operand)

    def visit_Name(self, node):
        return ('name', node.id)

    def visit_Constant(self, node):
        return ('const', node.value)

    def visit_Call(self, node):
        func = node.func.id
        args = [self.visit(arg) for arg in node.args]
        return ('call', func, args)
    
    def visit_Subscript(self, node):
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if self.operations == []:
            self.operations.append(('subscript', value, index))
        return ('subscript', value, index)

    def get_node_representation(self, node):
        if isinstance(node, ast.Name):
            return ('name', node.id)
        elif isinstance(node, ast.Constant):
            return ('const', node.value)
        elif isinstance(node, ast.Subscript):
            value = self.visit(node.value)
            index = self.visit(node.slice)
            return ('subscript', value, index)
        elif isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op = self.get_op_symbol(node.op)
            return (op, left, right)
        elif isinstance(node, ast.UnaryOp):
            op = self.get_op_symbol(node.op)
            operand = self.visit(node.operand)
            return ('unary', op, operand)
        elif isinstance(node, ast.Call):
            func = node.func.id
            args = [self.visit(arg) for arg in node.args]
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

def generate_placeholders_and_simplify(operations):
    placeholders = {}
    placeholder_count = 1
    
    def create_placeholder():
        nonlocal placeholder_count
        placeholder = f'x{placeholder_count}'
        placeholder_count += 1
        return placeholder
    
    def process_operation(op):
        if isinstance(op, tuple):
            if op[0] in ['name', 'const', 'subscript']:
                if op not in placeholders:
                    placeholders[op] = create_placeholder()
                return placeholders[op]
            elif op[0] in ['add', 'sub', 'mult', 'div']:
                left = process_operation(op[1])
                right = process_operation(op[2]) if len(op) > 2 else ""
                op_str = {'add': '+', 'sub': '-', 'mult': '*', 'div': '/'}[op[0]]
                if left == None and right == None:
                    return f"{op_str}"
                elif left == None:
                    return f"{op_str} {right}"
                elif right == None:
                    return f"{left} {op_str}"
                else:
                    return f"({left} {op_str} {right})"
                
            elif op[0] == 'unary':
                right = process_operation(op[2])
                return f"({op[1]} {right})"
            elif op[0] == 'call':
                args = ', '.join([process_operation(arg) for arg in op[2]])
                return f"{op[1]}({args})"
        return str(op)
    
    sympy_expressions = []

    for op_list in operations:
        if op_list:
            expr_str = ""
    
            for op in op_list:
                expr_str += process_operation(op)
            try:
                sympy_expr = sympify(expr_str, evaluate=False)
                sympy_expressions.append(sympy_expr)
            except (SympifyError, TypeError) as e:
                print(f"Error: Could not parse {expr_str} - {e}")
    print(expr_str)
    common_expr = simplify(Add(*sympy_expressions))
    
    return common_expr, placeholders
    for op_list in operations:
        for op in op_list:
            expr_str += process_operation(op)
        try:
            if expr_str:
                sympy_expr = sympify(expr_str, evaluate=False)
        except (SympifyError, TypeError) as e:
            print(f"Error: Could not parse {expr_str} - {e}")
        print(expr_str)
    common_expr = simplify(Add(*sympy_expressions))
    
    return common_expr, placeholders

def generate_minimal_representation(expressions):
    all_operations = [analyze_expression(expr) for expr in expressions]
    
    unique_operations = set()
    for ops in all_operations:
        unique_operations.update(ops)
    
    minimal_representation = list(unique_operations)
    
    return minimal_representation, all_operations

def generate_minimal_expression(expressions):
    sympy_expressions = []
    common_expressions=[]
    for expr in expressions:
        if expr:
            for operation in expr:
                sympy_expr_str = expr_to_sympy(operation)
                print(f"SymPy expression string: {sympy_expr_str}")
                try:
                    sympy_expr = sympify(sympy_expr_str, evaluate=False)
                    sympy_expressions.append(sympy_expr)
                except SympifyError as e:
                    print(f"SympifyError: Could not parse {sympy_expr_str} - {e}")
                except TypeError as e:
                    print(f"TypeError: Could not parse {sympy_expr_str} - {e}")
            
            if sympy_expressions:

                common_expr = Add(*sympy_expressions)
                common_expr = expand(common_expr)
                common_expr = factor(common_expr)
                common_expr = simplify(common_expr)
                common_expressions.append(common_expr)
            else: 
                common_expressions.append(0)
    return common_expressions

def expr_to_sympy(expr):
    if isinstance(expr, list) and len(expr) == 1:
        expr = expr[0]

    expr_str = ""
    if expr[0] == 'name':
        expr_str += expr[1]
    elif expr[0] == 'const':
        expr_str += str(expr[1])
    elif expr[0] in ['add', 'sub', 'mult', 'div']:
        left = expr_to_sympy(expr[1])
        right = expr_to_sympy(expr[2])
        op = {'add': '+', 'sub': '-', 'mult': '*', 'div': '/'}[expr[0]]
        expr_str += f"({left} {op} {right})"
    elif expr[0] == 'unary':
        expr_str += f"({expr[1]}{expr_to_sympy(expr[2])})"
    elif expr[0] == 'subscript':
        value = expr_to_sympy(expr[1])
        index = expr_to_sympy(expr[2])
        expr_str += f"{value}_{index}"
    elif expr[0] == 'call':
        args = ', '.join([expr_to_sympy(arg) for arg in expr[2]])
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

minimal_expression, placeholders = generate_placeholders_and_simplify(all_operations)


print(f"Minimal Expression: {minimal_expression}\n")
print(f"Placeholders: {placeholders}\n")
