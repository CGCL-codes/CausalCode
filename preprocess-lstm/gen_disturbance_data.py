

"""
description：
"""
import os
import random
import re
import shutil
from pycparser import c_parser
import random
import random
import string


def find_insert_locations(code: str) -> list:
    dead_code_locations = []

    for match in re.finditer(r'(\{[ \t\n]*\n)', code):
        line_number = code.count('\n', 0, match.start())
        column_number = match.start() - code.rfind('\n', 0, match.start())
        # struct
        if not re.search(r'struct\s+\w+\s*\{[^}]*' + re.escape(code[match.start():match.end()]) + r'[^}]*\}\s*\w',
                         code):
            dead_code_locations.append((line_number, column_number))

    # ';}’
    for match in re.finditer(r'(\;[ \t]*\})', code):
        line_number = code.count('\n', 0, match.start())
        column_number = match.start() - code.rfind('\n', 0, match.start())
        dead_code_locations.append((line_number, column_number))

    keywords = ['int', 'for', 'char', 'double', 'float', 'long', 'short', 'unsigned', 'signed', 'void', 'struct',
                'enum', 'if']
    for keyword in keywords:
        for match in re.finditer(r'(\{[ \t\n]*' + keyword + ')', code):
            line_number = code.count('\n', 0, match.start())
            column_number = match.start() - code.rfind('\n', 0, match.start())
            dead_code_locations.append((line_number, column_number))
    # struct
    for match in re.finditer(r'struct\s+\w+\s*\{[^}]*\}\s*;', c_code):
        start_line_number = c_code.count('\n', 0, match.start()) + 1
        end_line_number = c_code.count('\n', 0, match.end()) + 1
        dead_code_locations = [location for location in dead_code_locations if
                               not (start_line_number <= location[0] <= end_line_number)]
    return dead_code_locations


dead_code = [
    ";",
    "{ }",
    "printf ( \"\" ) ;",
    "if ( false ) ;",
    "if ( true ) { }",
    "if ( false ) ; else { }",
    "if ( 0 ) ;",
    "if ( false ) { int cnt = 0 ; for ( int i = 0 ; i < 123 ; i ++ ) cnt += 1 ; }"
    "for ( int i = 0 ; i < 100 ; i ++ ) break ;",
    "for ( int i = 0 ; i < 0 ; i ++ ) { }"
    "while ( false ) ;",
    "while ( 0 ) ;",
    "while ( true ) break ;",
    "for ( int i = 0 ; i < 10 ; i ++ ) { for ( int j = 0 ; j < 10 ; j ++ ) break ; break ; }",
    "do { } while ( false ) ;"]


def ins_dead_code(code_path, save_path):
    count = 0
    parser = c_parser.CParser()
    for root, dirs, files in os.walk(code_path):
        for file in files:
            relative_path = os.path.relpath(root, code_path)
            os.makedirs(os.path.join(save_path, relative_path), exist_ok=True)
            if os.path.exists(os.path.join(save_path, relative_path, file)):
                continue
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                c_code = f.read()
            insertion_points = find_insert_locations(c_code)
            c_code_lines = c_code.split('\n')

            if len(insertion_points) > 0:
                insertion_point = random.choice(insertion_points)
                dead_code_str = random.choice(dead_code)
                loop_count = 0
                while 'struct' in c_code_lines[insertion_point[0] - 1] or 'struct' in c_code_lines[
                    insertion_point[0]]:
                    insertion_point = random.choice(insertion_points)
                    loop_count += 1
                    if loop_count > 15:
                        break
                # 
                line_number, column_number = insertion_point
                c_code_lines[line_number] = c_code_lines[line_number][
                                            :column_number] + '\n' + dead_code_str + '\n' + \
                                            c_code_lines[line_number][column_number:]

                c_code_with_dead_code = '\n'.join(c_code_lines)
                try:
                    parser.parse(c_code_with_dead_code)
                    with open(os.path.join(save_path, relative_path, file), 'w', encoding='utf-8') as f:
                        f.write(c_code_with_dead_code)

                except Exception as e:
                    # print(os.path.join(data_root, file))
                    # print(c_code)
                    # print(c_code_with_dead_code)
                    print(e)
            else:
                print(os.path.join(root, file))
            if not os.path.exists(os.path.join(save_path, relative_path, file)):
                shutil.copy(os.path.join(root, file), os.path.join(save_path, relative_path, file))
                print(os.path.join(root, file))
                count += 1
    print(count)


__key_words__ = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__"]
__ops__ = ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
           ">>", "<<", "++", "--", "->", "&&", "||", "<=", ">=", "==", "!=", ";",
           "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>",
           ".", "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
__macros__ = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]  # <stdlib.h> macro
__special_ids__ = ["main",
                   "stdio", "cstdio", "stdio.h",  # <stdio.h> & <cstdio>
                   "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",  # <stdio.h> types & streams
                   "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush",  # <stdio.h> functions
                   "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                   "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                   "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                   "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                   "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                   "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                                                                     "stdlib", "cstdlib", "stdlib.h",

                   "size_t", "div_t", "ldiv_t", "lldiv_t",  # <stdlib.h> types
                   "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                   "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                   "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                   "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                   "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                   "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                   "mbstowcs", "wcstombs",
                   "string", "cstring", "string.h",  # <string.h> & <cstring>
                   "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",  # <string.h> functions
                   "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                   "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                   "strpbrk", "strstr", "strtok", "strxfrm",
                   "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",
                   # <string.h> extension functions
                   "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                   "iostream", "istream", "ostream", "fstream", "sstream",  # <iostream> family
                   "iomanip", "iosfwd",
                   "ios", "wios", "streamoff", "streampos", "wstreampos",  # <iostream> types
                   "streamsize", "cout", "cerr", "clog", "cin",
                   "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",  # <iostream> manipulators
                   "noshowbase", "showpoint", "noshowpoint", "showpos",
                   "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                   "left", "right", "internal", "dec", "oct", "hex", "fixed",
                   "scientific", "hexfloat", "defaultfloat", "width", "fill",
                   "precision", "endl", "ends", "flush", "ws", "showpoint",
                   "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",  # <math.h> functions
                   "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                   "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr", 'u', 'U', 'UU', 'uU', 'Uu']
forbidden_uid = __key_words__ + __ops__ + __macros__ + __special_ids__

import re


def find_replaceable_names(code: str) -> list:
    # 
    code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '', code)
    code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", '', code)
    # 、
    pattern = r'(?<![\w\\%])[_a-zA-Z][_a-zA-Z0-9]*(?![\w\\%])'
    #  findall 
    names = re.findall(pattern, code)
    # 、、
    names = [name for name in names if name not in forbidden_uid]
    # 
    names = list(set(names))
    return names


def change_token(old_name, new_name, code):
    pattern = r'\b{}\b'.format(old_name)
    new_code = re.sub(pattern, new_name, code)
    return new_code


def generate_random_tokens(names: list) -> dict:
    tokens = {}
    used_tokens = set()
    for name in names:
        #  token 
        min_len = max(1, len(name) - 1)
        max_len = len(name) + 1
        #  token
        while True:
            token_len = random.randint(min_len, max_len)
            #  token 
            first_char = random.choice(string.ascii_letters + '_')
            other_chars = ''.join(random.choices(string.ascii_letters + string.digits + '_', k=token_len - 1))
            token = first_char + other_chars
            if token not in forbidden_uid and token not in names and token not in used_tokens:
                break
        #  token 
        tokens[name] = token
        used_tokens.add(token)
    return tokens


def generate_format_token(names: list) -> dict:
    tokens = {}
    for i, name in enumerate(names):
        tokens[name] = 'token' + str(i)
    return tokens


if __name__ == '__main__':
    code_path = r'../data_raw/origin'
    save_path = r'../data_raw/token_format'  # token_dis
    os.makedirs(save_path, exist_ok=True)
    parser = c_parser.CParser()
    error_count = 0
    for root, dirs, files in os.walk(code_path):
        for file in files:
            relative_path = os.path.relpath(root, code_path)
            os.makedirs(os.path.join(save_path, relative_path), exist_ok=True)
            with open(os.path.join(root, file), 'r') as f:
                c_code = f.read()
            or_names = find_replaceable_names(c_code)
            chang_tokens = generate_format_token(or_names)
            for or_name in or_names:
                changed_code = change_token(or_name, chang_tokens[or_name], c_code)
                try:
                    parser.parse(changed_code)
                    c_code = changed_code
                except Exception as e:
                    error_count += 1
                    print(os.path.join(root, file))
                    print(c_code)
                    print(changed_code)
                    print(e)
            with open(os.path.join(save_path, relative_path, file), 'w') as f:
                f.write(c_code)
    print('error_count:', error_count)
