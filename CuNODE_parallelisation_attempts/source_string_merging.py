import re

def parse_function(function_str):
    # Split the function string into individual assignments
    assignments = function_str.strip().split('\n')
    return assignments

def extract_operands_and_results(assignments):
    variables = []
    constants = []
    results = []
    all_vars = set()
    variable_pattern = re.compile(r'\b[a-zA-Z_]\w*\b')
    constant_pattern = re.compile(r'\b\d+\b')
    
    for assignment in assignments:
        result_var, expression = assignment.split('=')
        result_var = result_var.strip()
        results.append(result_var)
        
        var_match = variable_pattern.findall(expression)
        const_match = constant_pattern.findall(expression)
        
        variables.append(var_match)
        constants.append(const_match)
        all_vars.update(var_match)
    
    all_vars = list(all_vars)
    var_indices = {var: i for i, var in enumerate(all_vars)}
    
    var_array = []
    for var_list in variables:
        var_array.append([var_indices[var] for var in var_list])
    
    return var_array, constants, results, all_vars, var_indices

def generate_single_function(assignments, variables, constants, results, all_vars, var_indices):
    function_body = ""
    for i, assignment in enumerate(assignments):
        result_var = results[i]
        expression = assignment.split('=')[1].strip()
        
        for var in all_vars:
            expression = expression.replace(var, f"variables[tx, {var_indices[var]}]")
        
        for j, const in enumerate(constants[i]):
            expression = expression.replace(const, f"constants[tx, {j}]")
        
        function_body += f"    results[tx, {i}] = {expression}\n"
    
    function_code = f"""
def single_function(tx, results, variables, constants):
{function_body}
"""
    return function_code

def main():
    # Example user function as a string
    user_function = """
a = b + c + d * f
d = g * h + b - c
"""

    # Step 1: Parse the user function
    assignments = parse_function(user_function)
    
    # Step 2: Extract operands and results
    var_array, constants, results, all_vars, var_indices = extract_operands_and_results(assignments)
    
    # Step 3: Generate the single function
    function_code = generate_single_function(assignments, var_array, constants, results, all_vars, var_indices)
    
    # Print generated function
    print(function_code)

if __name__ == "__main__":
    main()
