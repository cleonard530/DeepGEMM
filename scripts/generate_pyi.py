import re
from pathlib import Path


class BracketTracker:
    """
    Tracks nesting levels of various brackets in C++ code:
      - () → paren
      - [] → bracket
      - {} → brace
      - <> → angle (treated as template brackets only at top level)
    Provides is_top_level() to check if currently outside all brackets.
    """
    def __init__(self):
        self.paren = 0      # ()
        self.bracket = 0    # []
        self.brace = 0      # {}
        self.angle = 0      # <>

    def update(self, char: str):
        """
        Update internal counters based on the given character.
        """
        if char == '(':
            self.paren += 1
        elif char == ')':
            self.paren -= 1
        elif char == '[':
            self.bracket += 1
        elif char == ']':
            self.bracket -= 1
        elif char == '{':
            self.brace += 1
        elif char == '}':
            self.brace -= 1
        # Angle brackets < > are only treated as template delimiters
        # when not inside (), [], or {}
        elif char == '<' and self._in_top_level_of_other_brackets():
            self.angle += 1
        elif char == '>' and self.angle > 0 and self._in_top_level_of_other_brackets():
            self.angle -= 1

    def _in_top_level_of_other_brackets(self):
        """
        Check if not inside parentheses, square brackets, or braces (for correct template bracket recognition).
        """
        return self.paren == 0 and self.bracket == 0 and self.brace == 0

    def is_top_level(self):
        """
        Check if completely at top level (all bracket counters are zero).
        """
        return (self.paren == 0 and
                self.bracket == 0 and
                self.brace == 0 and
                self.angle == 0)


def extract_torch_op_name(schema: str) -> str:
    """Extract the operator name from a TORCH schema string."""
    paren_pos = schema.find('(')
    if paren_pos == -1:
        return schema.strip()
    return schema[:paren_pos].strip()


def extract_m_def_statements(root_path):
    """
    Scan all C++ files under root_path and extract all m.def(...) statements.
    Supports multi-line m.def(...) calls.
    """
    results = []
    extensions = {'.hpp', '.cpp', '.h', '.cc'}

    for file_path in Path(root_path).rglob('*'):
        if file_path.suffix.lower() not in extensions:
            continue
        if not file_path.is_file():
            continue

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f'Failed to read file {file_path}: {e}')
            continue

        m_def_list = []
        lines = content.splitlines(keepends=True)
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'm.def(' in line:
                # Found a potential starting line
                # Check if it's a comment
                stripped = line.lstrip()
                if stripped.startswith('//') or stripped.startswith('/*'):
                    i += 1
                    continue

                # Try to match the complete m.def(...) call
                paren_count = 0
                j = i
                found_start = False
                while j < len(lines):
                    current_line = lines[j]
                    for k, char in enumerate(current_line):
                        if char == '(':
                            if not found_start and re.search(r'm\.def\s*\(', current_line[:k+1]):
                                found_start = True
                            if found_start:
                                paren_count += 1
                        elif char == ')':
                            if found_start:
                                paren_count -= 1
                                if paren_count == 0:
                                    # Found complete statement
                                    full_stmt = ''.join(lines[i:j+1]).rstrip()
                                    m_def_list.append(full_stmt)
                                    i = j
                                    break
                    if paren_count <= 0 and found_start:
                        break
                    j += 1
            i += 1

        if m_def_list:
            results.append({
                'file': str(file_path),
                'm_def_statements': m_def_list
            })

    return results


def parse_m_def_statement(m_def_str):
    """
    Parse a TORCH_LIBRARY m.def(...) statement.

    DeepGEMM registers ops via TORCH_LIBRARY_FRAGMENT, so the first argument is
    always a schema string such as "fp8_fp4_gemm_nt(Tensor a, ...) -> ()".
    """
    # Extract top-level arguments
    start = m_def_str.find('m.def(')
    if start == -1:
        raise ValueError(f'[{m_def_str}] Could not find m.def start position')

    paren_count = 0
    content_start = start + len('m.def(')
    content_end = -1
    for i in range(content_start, len(m_def_str)):
        ch = m_def_str[i]
        if ch == '(':
            paren_count += 1
        elif ch == ')':
            if paren_count == 0:
                content_end = i
                break
            else:
                paren_count -= 1
    if content_end == -1:
        raise ValueError(f'[{m_def_str}] m.def parentheses not closed')

    args_content = m_def_str[content_start:content_end]

    # Split arguments using BracketTracker
    args_list = []
    current = []
    tracker = BracketTracker()

    for ch in args_content:
        if ch in '()[]{}<>':
            tracker.update(ch)
        if ch == ',' and tracker.is_top_level():
            args_list.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        args_list.append(''.join(current).strip())

    if not args_list:
        raise ValueError(f'[{m_def_str}] m.def has no arguments')

    # Extract operator name from the TORCH schema string
    first = args_list[0].strip()
    str_match = re.match(r'^"([^"\\]*(?:\\.[^"\\]*)*)"', first)
    if not str_match:
        raise ValueError(f'[{m_def_str}] m.def first argument should be a string literal')

    schema = str_match.group(1)
    return {
        'python_function_name': extract_torch_op_name(schema),
        'schema': schema,
    }


def generate_pyi_function(item_entry):
    """
    Generate a .pyi stub for one registered op.

    Typed stubs require parsing the TORCH schema in item_entry['parsed']['schema'].
    Until then, emit a generic signature that matches deep_gemm._C wrappers.
    """
    py_name = item_entry['parsed']['python_function_name']
    return f'def {py_name}(*args, **kwargs) -> Any: ...'


def generate_pyi_file_content(enhanced_results, module_name: str = 'my_module'):
    function_decls = []
    has_torch = False

    for item in enhanced_results:
        for stmt in item['m_def_statements']:
            try:
                decl = generate_pyi_function(stmt)
                function_decls.append(decl)
                if 'torch.Tensor' in decl:
                    has_torch = True
            except Exception as e:
                func_name = stmt['parsed'].get('python_function_name', 'unknown')
                function_decls.append(f'# ERROR: failed to generate stub for {func_name}: {e}')

    lines = [
        f'# Stubs for module: {module_name}',
        '',
        'from typing import Any',
    ]
    if has_torch:
        lines.append('import torch')
    lines.extend(['', ''])

    for decl in function_decls:
        lines.extend([decl, '', ''])

    return '\n'.join(lines)


def generate_pyi_file(name, root, output_dir='.'):
    results = extract_m_def_statements(root)

    enhanced_results = []
    for item in results:
        statements = []
        for stmt in item['m_def_statements']:
            statements.append({
                'raw': stmt,
                'parsed': parse_m_def_statement(stmt),
            })
        enhanced_results.append({'m_def_statements': statements})

    pyi_content = generate_pyi_file_content(enhanced_results, module_name=name)

    output_path = Path(output_dir) / f'{name}.pyi'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pyi_content)

    print(f'.pyi file generated: {output_path}')
