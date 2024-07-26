import os
import subprocess
import sys

def run_pyside6_uic(ui_file):
    py_file = ui_file.replace('.ui', '.py')
    command = ['pyside6-uic', ui_file, '-o', py_file]
    subprocess.run(command, check=True)
    return py_file

def replace_in_file(file_path, old_string, new_string):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content = content.replace(old_string, new_string)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def process_files_in_directory(directory, old_string, new_string):
    script_name = os.path.basename(__file__)

    for root, _, files in os.walk(directory):
        for file in files:
            if file == script_name:
                    continue
            if file.endswith('.ui'):
                ui_file_path = os.path.join(root, file)
                py_file_path = run_pyside6_uic(ui_file_path)
                print(f'Generated {py_file_path} from {ui_file_path}')
                replace_in_file(py_file_path, old_string, new_string)

            elif file.endswith('.py'):
                py_file_path = os.path.join(root, file)
                replace_in_file(py_file_path, old_string, new_string)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory_to_search = sys.argv[1]
    else:
        directory_to_search = os.getcwd()

    old_string = "PySide6"
    new_string = "qtpy"

    process_files_in_directory(directory_to_search, old_string, new_string)
