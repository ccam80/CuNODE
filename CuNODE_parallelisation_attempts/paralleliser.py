import ast
import inspect
import re

def tokenize(expression):
    tokens = re.findall(r'constants\[\d+\]|state\[\d+\]|\b[a-zA-Z_]\w*\b|\d+\.\d+|\d+|[\+\-\*/\(\)\[\]]', expression)
    return tokens

def parse_tokens(tokens):
    placeholders = {}
    parsed_expression = []
    placeholder_count = 1

    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if re.match(r'constants\[\d+\]|state\[\d+\]|\b[a-zA-Z_]\w*\b|\d+\.\d+|\d+', token):
            placeholder = f'x{placeholder_count}'
            placeholders[placeholder] = token
            parsed_expression.append(placeholder)
            placeholder_count += 1
        elif token == '-':
                parsed_expression.append('+')
                next_token = tokens[i + 1]
                placeholders[f'x{placeholder_count}'] = f'-{next_token}'
                parsed_expression.append(f'x{placeholder_count}')
                placeholder_count += 1
                i += 1  # Skip the next token
        else:
            parsed_expression.append(token)
        
        i += 1
    
    return parsed_expression, placeholders

def group_operations(expression):
    groups = []
    current_group = []

    for token in expression:
        if token in '+-':
            if current_group:
                groups.append(current_group)
                current_group = []
            groups.append([token])
        else:
            current_group.append(token)
    
    if current_group:
        groups.append(current_group)

    return groups

def normalize_expression(groups):
    normalized_groups = []
    
    for group in groups:
        if group[0] in '+-':
            normalized_groups.append(group)
        else:
            group.sort()
            normalized_groups.append(group)
    
    normalized_expression = ''.join([''.join(group) for group in normalized_groups])
    return normalized_expression

def quantify_groups(groups):
    quantified_groups = []

    for group in groups:
        if group[0] in '+-':
            quantified_groups.append(group)
        else:
            group_length = len([token for token in group if re.match(r'x\d+', token)])
            quantified_groups.append(('*', group_length, group))
    
    return quantified_groups

def find_minimal_representation(expressions):
    all_quantified_groups = []
    all_placeholders = []
    group_sources = []

    for expr in expressions:
        tokens = tokenize(expr)
        parsed_expression, placeholders = parse_tokens(tokens)
        groups = group_operations(parsed_expression)
        quantified_groups = quantify_groups(groups)
        
        all_quantified_groups.append(quantified_groups)
        all_placeholders.append(placeholders)
    
    minimal_representation = []
    ordered_expressions = all_quantified_groups.copy()
    ordered_placeholders = all_placeholders.copy()
    
    while any(all_quantified_groups):
        ordered_expressions.append([])
        max_group_size = max(group[1] for expr in all_quantified_groups for group in expr if isinstance(group, tuple))

        max_overall_group = None
        for expr_index, expr in enumerate(all_quantified_groups):
            max_expr_group = None
            for group in expr:
                if isinstance(group, tuple):
                    if max_expr_group is None or group[1] > max_expr_group[1]:
                        max_expr_group = group
                elif group == ['+']:
                    expr.remove(group)
                    

                if max_expr_group and max_expr_group[1] == max_group_size:
                    max_overall_group = max_expr_group
                    group_sources.append(expr_index)
                    break

            if max_expr_group:
                ordered_expressions[expr_index].append(max_expr_group)
                expr.remove(max_expr_group)

        if max_overall_group:
            minimal_representation.append(max_overall_group)
        
        all_quantified_groups = [expr for expr in all_quantified_groups if any(isinstance(group, tuple) for group in expr)]
    
    # Assign new placeholders to the minimal representation
    new_placeholders = {}
    placeholder_count = 1
    for group in minimal_representation:
        if isinstance(group, tuple):
            for multiplicand in group[2]:
                if multiplicand != '*' and multiplicand not in new_placeholders:
                    new_placeholders[multiplicand] = f'x{placeholder_count}'
                    placeholder_count += 1

    # Update minimal representation with new placeholders
    minimal_expression_parts = []
    for group in minimal_representation:
        if isinstance(group, tuple):
            minimal_expression_parts.append('*'.join(new_placeholders[multiplicand] for multiplicand in group[2]))
        else:
            minimal_expression_parts.append(group[0])

    minimal_expression = ' + '.join(minimal_expression_parts)
    print(f"Minimal Expression: {minimal_expression}\n")

    # Reassign placeholders for each original expression
    updated_placeholders = []
    for i, expr in enumerate(expressions):
        updated_expr_placeholders = []
        for group_index, group in enumerate(minimal_representation):
            if isinstance(group, tuple):
                source_index = group_sources[group_index]
                if source_index == i:
                    updated_expr_placeholders.extend(new_placeholders[multiplicand] for multiplicand in group[2])
                else:
                    updated_expr_placeholders.extend('1' for _ in group[2])
        updated_placeholders.append(updated_expr_placeholders)
    
    return minimal_representation, updated_placeholders

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


    
def dxdtfunc(outarray, state, constants, t):
    ref = clamp(cos(constants[13] * t) * constants[9], constants[10])
    control = linear_control_eq(constants[11], constants[12], state[4], constants[10])

    outarray[0] = state[1]
    outarray[1] = (-state[0] - constants[3] * state[1] + constants[0] * state[2] + constants[7] * ref)
    outarray[2] = (-constants[1] * state[2] + constants[2] * state[3] * state[3])
    outarray[3] = (-constants[6] * state[3] + constants[6] * control)
    outarray[4] = (-constants[5] * state[4] + constants[8] * state[1])



expressions = extract_expressions_from_function(dxdtfunc)
minimal_representation, updated_placeholders = find_minimal_representation(expressions)

for i, expr in enumerate(expressions):
    print(f"Original: {expr}")
    print(f"Groups: {minimal_representation}")
    print(f"Values: {updated_placeholders[i]}")
    print()