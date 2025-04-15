import json
import os
import re
from pycparser import c_parser, preprocess_file
import ast

parser = c_parser.CParser()


def remove_comment(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def check_syntax(code, parser):
    try:
        preprocessed_code = preprocess_file(code, cpp_args='-E')
        ast = parser.parse(preprocessed_code)
    except Exception as e:
        print(e)
        print(preprocessed_code)
        return False
    return True


if __name__ == '__main__':

    root = '../code_defect/data_raw/'
    with open(root + 'function.json', 'r') as f:
        data = json.load(f)

    # os.path.join(root,'origin')
    import shutil

    # 
    if os.path.exists(os.path.join(root, 'origin')):
        shutil.rmtree(os.path.join(root, 'origin'))
    # th.join(root,'origin')
    os.makedirs(os.path.join(root, 'origin'), exist_ok=True)

    for index, item in enumerate(data):
        target = item['target']
        func = item['func']
        func = re.sub(r'/\*.*?\*/', '', func, flags=re.DOTALL)
        func = re.sub(r'//.*?\n', '\n', func)
        directory = str(target)
        os.makedirs(os.path.join(root, 'origin', directory), exist_ok=True)
        filename = f'{index + 1}.c'
        with open(os.path.join(root, 'origin', directory, filename), 'w') as f:
            f.write(func)
    print('done')
