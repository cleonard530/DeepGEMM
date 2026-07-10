"""
Generate deep_gemm/_C.pyi from TORCH_LIBRARY schemas and deep_gemm/_C.py wrappers.

Pipeline (one op at a time, e.g. fp8_fp4_gemm_nt):

  1. extract_m_def_statements(csrc/)
       Scan csrc/apis/*.hpp for m.def("...schema...") registrations.
       Returns a flat list of raw m.def(...) statement strings.

  2. parse_m_def_statement(stmt) -> parse_torch_schema(schema)
       Split the schema string into structured params (flat, as registered in C++).
       Example output:
         name='fp8_fp4_gemm_nt'
         parameters=[
           {'name': 'a',   'py_type': 'torch.Tensor', 'default': None},
           {'name': 'sfa', 'py_type': 'torch.Tensor', 'default': None},
           {'name': 'b',   'py_type': 'torch.Tensor', 'default': None},
           ...
         ]

  3. parse_c_py_metadata(deep_gemm/_C.py)
       Read wrapper defaults from the Python shim (including names exported via
       globals().update aliases, so fp8_gemm_nt picks up fp8_fp4_gemm_nt defaults).
       Example:
         defaults['fp8_einsum']['recipe'] = '(1, 128, 128)'

  4. adjust_for_c_py_wrapper(name, parameters, wrapper_defaults)
       Reshape flat schema params to match the public _C.py API.
       Example: (a, sfa), (b, sfb) -> a: tuple[Tensor, Tensor], b: tuple[...]

  5. apply_wrapper_defaults(name, parameters, wrapper_defaults)
       Overlay defaults from _C.py where they differ from the TORCH schema.
       Example: recipe=None in schema -> recipe=(1, 128, 128) from fp8_einsum()

  6. generate_pyi_function(...) -> str
       Render one stub def line block for the .pyi file.

  7. generate_pyi_file_content(...)
       Emit one stub per TORCH_LIBRARY op found in csrc/.
"""
import ast
import re
from pathlib import Path

_TENSOR_PAIR = 'tuple[torch.Tensor, torch.Tensor]'
_Q_TUPLE = 'tuple[torch.Tensor, Optional[torch.Tensor]]'


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


def split_top_level_commas(value: str) -> list[str]:
    """Split a string on top-level commas.

    Example:
      "Tensor a, Tensor? c=None, int[]? recipe=None"
        -> ["Tensor a", "Tensor? c=None", "int[]? recipe=None"]
    Commas inside brackets/parens (e.g. Tensor(d!) d) are ignored.
    """
    parts = []
    current = []
    tracker = BracketTracker()
    for ch in value:
        if ch in '()[]{}<>':
            tracker.update(ch)
        if ch == ',' and tracker.is_top_level():
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return parts


def find_top_level_equals(value: str) -> int:
    """Return index of top-level '=' in a schema argument, or -1."""
    tracker = BracketTracker()
    for i, ch in enumerate(value):
        if ch in '()[]{}<>':
            tracker.update(ch)
        elif ch == '=' and tracker.is_top_level():
            return i
    return -1


def schema_type_to_python(type_str: str) -> str:
    """Map a TORCH_LIBRARY schema type to a Python type annotation string.

    Examples:
      "Tensor"     -> "torch.Tensor"
      "Tensor?"    -> "Optional[torch.Tensor]"
      "int[]"      -> "list[int]"
      "int[]?"     -> "Optional[list[int]]"
    """
    type_str = type_str.strip()
    optional = type_str.endswith('?')
    if optional:
        type_str = type_str[:-1].strip()

    if type_str.startswith('Tensor'):
        py_type = 'torch.Tensor'
    elif type_str == 'int':
        py_type = 'int'
    elif type_str == 'bool':
        py_type = 'bool'
    elif type_str == 'float':
        py_type = 'float'
    elif type_str == 'str':
        py_type = 'str'
    elif type_str == 'int[]':
        py_type = 'list[int]'
    else:
        print(f'Warning: unrecognized schema type {type_str!r}, using Any')
        py_type = 'Any'

    if optional:
        return f'Optional[{py_type}]'
    return py_type


def schema_return_to_python(return_str: str) -> str:
    """Map a TORCH_LIBRARY return type to a Python annotation."""
    return_str = return_str.strip()
    if return_str == '()':
        return 'None'
    if return_str in {'int', 'bool', 'float', 'str', 'Tensor'}:
        return {
            'int': 'int',
            'bool': 'bool',
            'float': 'float',
            'str': 'str',
            'Tensor': 'torch.Tensor',
        }[return_str]
    if return_str.startswith('(') and return_str.endswith(')'):
        inner = return_str[1:-1].strip()
        if not inner:
            return 'tuple[()]'
        parts = split_top_level_commas(inner)
        py_parts = [schema_return_to_python(part) for part in parts]
        return f'tuple[{", ".join(py_parts)}]'
    print(f'Warning: unrecognized schema return type {return_str!r}, using Any')
    return 'Any'


def schema_default_to_python(default_str: str) -> str:
    """Convert a TORCH schema default literal to a Python expression string."""
    default_str = default_str.strip()
    if default_str in {'None', 'True', 'False'}:
        return default_str
    if (default_str.startswith("'") and default_str.endswith("'")) or (
            default_str.startswith('"') and default_str.endswith('"')):
        return default_str
    if re.match(r'^[+-]?\d+$', default_str):
        return default_str
    if re.match(r'^[+-]?\d*\.\d+([eE][+-]?\d+)?$', default_str):
        return default_str
    print(f'Warning: unrecognized schema default {default_str!r}, using None')
    return 'None'


def parse_schema_arg(arg_str: str) -> dict:
    """Parse one TORCH schema argument such as 'Tensor? c=None'.

    Example:
      "str compiled_dims='nk'"
        -> {'name': 'compiled_dims', 'py_type': 'str', 'default': "'nk'"}
    """
    arg_str = arg_str.strip()
    if not arg_str:
        raise ValueError('empty schema argument')

    default = None
    eq_pos = find_top_level_equals(arg_str)
    if eq_pos != -1:
        default = schema_default_to_python(arg_str[eq_pos + 1:].strip())
        arg_str = arg_str[:eq_pos].strip()

    match = re.match(r'^(.+?)\s+([a-zA-Z_][a-zA-Z0-9_]*)$', arg_str)
    if not match:
        raise ValueError(f'could not parse schema argument: {arg_str!r}')
    return {
        'name': match.group(2),
        'py_type': schema_type_to_python(match.group(1)),
        'default': default,
    }


def parse_torch_schema(schema: str) -> dict:
    """Parse a TORCH_LIBRARY schema into name, parameters, and return type.

    Example input:
      "fp8_fp4_gemm_nt(Tensor a, Tensor sfa, Tensor b, Tensor sfb, "
      "Tensor(d!) d, Tensor? c=None, str compiled_dims='nk') -> ()"

    Example output (abbreviated):
      {
        'python_function_name': 'fp8_fp4_gemm_nt',
        'return_type': 'None',
        'parameters': [
          {'name': 'a', 'py_type': 'torch.Tensor', 'default': None},
          {'name': 'sfa', 'py_type': 'torch.Tensor', 'default': None},
          ...
        ],
      }
    """
    arrow = schema.rfind(' -> ')
    if arrow == -1:
        raise ValueError(f'schema missing return type: {schema!r}')

    signature = schema[:arrow].strip()
    return_type = schema_return_to_python(schema[arrow + 4:].strip())

    open_paren = signature.find('(')
    if open_paren == -1:
        raise ValueError(f'schema missing argument list: {schema!r}')

    name = signature[:open_paren].strip()
    paren_depth = 0
    close_paren = -1
    for i in range(open_paren, len(signature)):
        if signature[i] == '(':
            paren_depth += 1
        elif signature[i] == ')':
            paren_depth -= 1
            if paren_depth == 0:
                close_paren = i
                break
    if close_paren == -1:
        raise ValueError(f'unclosed argument list in schema: {schema!r}')

    args_blob = signature[open_paren + 1:close_paren].strip()
    parameters = []
    if args_blob:
        for arg in split_top_level_commas(args_blob):
            parameters.append(parse_schema_arg(arg))

    return {
        'python_function_name': name,
        'parameters': parameters,
        'return_type': return_type,
    }


def _merge_named_pairs(parameters: list[dict], pairs: tuple[tuple[str, str], ...]) -> list[dict]:
    """Replace (left, right) arg pairs with a single tuple-typed parameter.

    Example: pairs=(('a', 'sfa'), ('b', 'sfb'))
      [a, sfa, b, sfb, d, ...]  ->  [a: tuple[Tensor, Tensor], b: tuple[...], d, ...]
    """
    drop = {right for left, right in pairs}
    merged_left = {left for left, _ in pairs}
    out = []
    for param in parameters:
        if param['name'] in drop:
            continue
        if param['name'] in merged_left:
            out.append({
                'name': param['name'],
                'py_type': _TENSOR_PAIR,
                'default': None,
            })
            continue
        out.append(dict(param))
    return out


def _is_tensor_schema_param(param: dict) -> bool:
    py_type = param['py_type']
    return py_type in {'torch.Tensor', 'Optional[torch.Tensor]'}


def _is_tensor_scale_factor_pair(base_name: str, sf_name: str) -> bool:
    if base_name == 'a' and sf_name == 'sfa':
        return True
    if base_name == 'b' and sf_name == 'sfb':
        return True
    return sf_name == f'{base_name}_sf'


def detect_tensor_sf_pairs(parameters: list[dict]) -> list[tuple[str, str]]:
    """Detect consecutive (tensor, scale_factor) arg pairs in a TORCH schema.

    Examples (flat schema params from step 2):
      [a, sfa, b, sfb, ...]           -> [('a', 'sfa'), ('b', 'sfb')]
      [kv, kv_sf, weights, ...]       -> [('kv', 'kv_sf')]
      [l1_weights, l1_weights_sf, ...] -> [('l1_weights', 'l1_weights_sf')]
    """
    pairs = []
    i = 0
    while i < len(parameters) - 1:
        left, right = parameters[i], parameters[i + 1]
        if (
            _is_tensor_schema_param(left)
            and _is_tensor_schema_param(right)
            and _is_tensor_scale_factor_pair(left['name'], right['name'])
        ):
            pairs.append((left['name'], right['name']))
            i += 2
        else:
            i += 1
    return pairs


def _apply_q_qsf_merge(parameters: list[dict]) -> list[dict]:
    """Merge optional q_sf into q for attention wrappers that accept either form.

    Example schema: q, q_sf, kv, kv_sf, ...
      -> q: tuple[Tensor, Optional[Tensor]], kv, kv_sf, ...
    (q_sf is dropped; kv/kv_sf merging happens separately via detect_tensor_sf_pairs.)
    """
    if not any(param['name'] == 'q_sf' for param in parameters):
        return [dict(param) for param in parameters]

    out = []
    for param in parameters:
        if param['name'] == 'q_sf':
            continue
        param = dict(param)
        if param['name'] == 'q':
            param['py_type'] = _Q_TUPLE
        out.append(param)
    return out


def _maybe_widen_int_list_value_param(parameters: list[dict]) -> None:
    """Single int[] value param in a Python wrapper usually accepts int | list[int]."""
    if len(parameters) == 1 and parameters[0]['name'] == 'value':
        if parameters[0]['py_type'] == 'list[int]':
            parameters[0]['py_type'] = 'int | list[int]'


def _promote_int_list_tuple_types(op_name: str, parameters: list[dict]) -> None:
    """Promote int[] schema params to fixed-size tuples matching the public _C.py API.

    TORCH schemas use int[] for C++ list/variant conversions; callers pass tuples.
    """
    for param in parameters:
        name = param['name']
        py_type = param['py_type']

        if name == 'head_splits' and py_type == 'list[int]':
            param['py_type'] = 'tuple[int, int, int]'
        elif name == 'recipe_a' and py_type == 'Optional[list[int]]':
            param['py_type'] = 'Optional[tuple[int, int]]'
        elif name == 'recipe_b' and py_type == 'Optional[list[int]]':
            param['py_type'] = 'Optional[tuple[int, int]]'
        elif name == 'recipe' and op_name == 'transform_sf_into_required_layout':
            if py_type == 'list[int]':
                param['py_type'] = 'tuple[int, int] | tuple[int, int, int]'
        elif name == 'recipe':
            if py_type == 'list[int]':
                param['py_type'] = 'tuple[int, int, int]'
            elif py_type == 'Optional[list[int]]':
                param['py_type'] = 'Optional[tuple[int, int, int]]'


def adjust_for_c_py_wrapper(
    name: str,
    parameters: list[dict],
    wrapper_defaults: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """
    Adjust parsed schema parameters to match deep_gemm._C Python wrappers.

    TORCH_LIBRARY registers flat tensor/scales args; _C.py preserves the legacy
    pybind API by accepting (tensor, scale_factor) tuples for many kernels.

    Example transformation for fp8_fp4_gemm_nt:
      schema:  a, sfa, b, sfb, d, c=None, recipe=None, compiled_dims='nk', ...
      stub:    a: tuple[Tensor, Tensor], b: tuple[Tensor, Tensor], d, c=None, ...
    """
    parameters = _apply_q_qsf_merge(parameters)

    pairs = detect_tensor_sf_pairs(parameters)
    if pairs:
        parameters = _merge_named_pairs(parameters, tuple(pairs))

    for param in parameters:
        if param['name'] == 'logits_dtype':
            param['py_type'] = 'torch.dtype'

    _maybe_widen_int_list_value_param(parameters)
    _promote_int_list_tuple_types(name, parameters)

    return parameters


def sanitize_param_name(name: str) -> str:
    if name in {'def', 'class', 'from', 'import', 'None', 'True', 'False'}:
        return f'{name}_'
    return name


def format_ast_default(node: ast.AST) -> str:
    """Convert an AST default value node to a Python expression string for stubs."""
    if isinstance(node, ast.Constant):
        if node.value is None:
            return 'None'
        if isinstance(node.value, bool):
            return 'True' if node.value else 'False'
        if isinstance(node.value, str):
            return f'"{node.value}"'
        if isinstance(node.value, (int, float)):
            return repr(node.value)
    if isinstance(node, ast.Tuple):
        elts = ', '.join(format_ast_default(element) for element in node.elts)
        return f'({elts})'
    if isinstance(node, ast.List):
        elts = ', '.join(format_ast_default(element) for element in node.elts)
        return f'[{elts}]'
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f'-{format_ast_default(node.operand)}'
    return ast.unparse(node)


def extract_function_defaults(func_def: ast.FunctionDef) -> dict[str, str]:
    """Extract {param_name: default_expr} from a Python function definition."""
    defaults: dict[str, str] = {}
    args = func_def.args
    pos_args = args.args
    if args.defaults:
        first_default_idx = len(pos_args) - len(args.defaults)
        for idx, default_node in enumerate(args.defaults):
            defaults[pos_args[first_default_idx + idx].arg] = format_ast_default(default_node)
    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        if default_node is not None:
            defaults[arg.arg] = format_ast_default(default_node)
    return defaults


def _is_globals_update_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != 'update':
        return False
    base = node.func.value
    if isinstance(base, ast.Name):
        return base.id == 'globals'
    if isinstance(base, ast.Call) and isinstance(base.func, ast.Name):
        return base.func.id == 'globals'
    return False


def parse_c_py_metadata(c_py_path: Path) -> dict[str, dict[str, str]]:
    """Parse wrapper defaults from deep_gemm/_C.py.

    Walks the AST (does not import or execute _C.py).

    Defaults example — from:
      def fp8_einsum(..., recipe=(1, 128, 128)):
    produces:
      defaults['fp8_einsum']['recipe'] = '(1, 128, 128)'

    Also copies defaults onto legacy alias names from globals().update(...) so
    wrapper defaults apply when the public name differs from the TORCH op name.
    """
    source = c_py_path.read_text(encoding='utf-8')
    module = ast.parse(source, filename=str(c_py_path))

    func_defaults: dict[str, dict[str, str]] = {}

    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            func_defaults[node.name] = extract_function_defaults(node)

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not _is_globals_update_call(node):
            continue
        if not node.args or not isinstance(node.args[0], ast.Dict):
            continue
        alias_dict = node.args[0]
        for key_node, value_node in zip(alias_dict.keys, alias_dict.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            alias_name = key_node.value
            if isinstance(value_node, ast.Name):
                if value_node.id in func_defaults:
                    func_defaults[alias_name] = func_defaults[value_node.id]

    return func_defaults


def extract_m_def_statements(root_path) -> list[str]:
    """
    Scan all C++ files under root_path and extract all m.def(...) statements.

    Returns a flat list of raw statement strings (one per registration found).
    Supports multi-line m.def(...) calls. This is pipeline step 1.

    Example match in gemm.hpp:
      m.def(
          "fp8_fp4_gemm_nt(Tensor a, Tensor sfa, ...) -> ()");
    """
    statements = []
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
            statements.extend(m_def_list)

    return statements


def parse_m_def_statement(m_def_str):
    """
    Parse a TORCH_LIBRARY m.def(...) statement (pipeline step 2).

    DeepGEMM registers ops via TORCH_LIBRARY_FRAGMENT; the first m.def argument
    is always a schema string. Extra args like DEEP_GEMM_IMPL(...) are ignored.

    Example input:
      m.def("bf16_gemm_nt(Tensor a, Tensor b, Tensor(d!) d, "
            "Tensor? c=None, str compiled_dims='nk') -> ()",
            DEEP_GEMM_IMPL(bf16_gemm_nt));

    Delegates to parse_torch_schema() on the first string literal.
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
    args_list = split_top_level_commas(args_content)

    if not args_list:
        raise ValueError(f'[{m_def_str}] m.def has no arguments')

    # Extract operator schema from the first string literal
    first = args_list[0].strip()
    str_match = re.match(r'^"([^"\\]*(?:\\.[^"\\]*)*)"', first)
    if not str_match:
        raise ValueError(f'[{m_def_str}] m.def first argument should be a string literal')

    return parse_torch_schema(str_match.group(1))


def apply_wrapper_defaults(name: str, parameters: list[dict], wrapper_defaults: dict[str, dict[str, str]]) -> list[dict]:
    """Overlay public API defaults from deep_gemm/_C.py onto schema-derived parameters.

    Schema defaults come from the TORCH registration string; wrapper defaults reflect
    what callers actually get from _C.py.

    Example for fp8_einsum:
      schema default:  recipe=None
      wrapper default: recipe=(1, 128, 128)   # from def fp8_einsum(..., recipe=(1, 128, 128))
      stub result:     recipe: tuple[int, int, int] = (1, 128, 128)
    """
    by_name = wrapper_defaults.get(name, {})
    if not by_name:
        return parameters

    out = []
    for param in parameters:
        param = dict(param)
        if param['name'] in by_name:
            param['default'] = by_name[param['name']]
        out.append(param)
    return out


def generate_pyi_function(parsed, wrapper_defaults=None):
    """Generate a typed .pyi stub for one registered op (pipeline steps 4-6).

    Example final output for fp8_fp4_gemm_nt:
      def fp8_fp4_gemm_nt(
          a: tuple[torch.Tensor, torch.Tensor],
          b: tuple[torch.Tensor, torch.Tensor],
          d: torch.Tensor,
          c: Optional[torch.Tensor] = None,
          ...
      ) -> None: ...
    """
    py_name = parsed['python_function_name']
    parameters = adjust_for_c_py_wrapper(
        py_name,
        parsed['parameters'],
        wrapper_defaults=wrapper_defaults,
    )
    if wrapper_defaults:
        parameters = apply_wrapper_defaults(py_name, parameters, wrapper_defaults)
    return_type = parsed['return_type']

    param_lines = []
    for param in parameters:
        name = sanitize_param_name(param['name'])
        if param['default'] is not None:
            param_lines.append(f'    {name}: {param["py_type"]} = {param["default"]}')
        else:
            param_lines.append(f'    {name}: {param["py_type"]}')

    if param_lines:
        params_block = ',\n'.join(param_lines)
        return f'def {py_name}(\n{params_block}\n) -> {return_type}: ...'
    return f'def {py_name}() -> {return_type}: ...'


def generate_pyi_file_content(
    parsed_ops,
    module_name: str = 'my_module',
    wrapper_defaults=None,
):
    """Assemble the full .pyi file from all parsed ops (pipeline step 7).

    parsed_ops: list of dicts returned by parse_m_def_statement / parse_torch_schema.

    - One stub per TORCH_LIBRARY op found in csrc/
    """
    decls = []

    for parsed in parsed_ops:
        name = parsed['python_function_name']
        try:
            decl = generate_pyi_function(parsed, wrapper_defaults=wrapper_defaults)
        except Exception as e:
            decl = f'# ERROR: failed to generate stub for {name}: {e}'
        decls.append(decl)

    lines = [
        f'# Stubs for module: {module_name}',
        '',
        'from typing import Any, Optional',
        'import torch',
        '',
    ]

    for decl in decls:
        lines.extend([decl, '', ''])

    return '\n'.join(lines)


def generate_pyi_file(name, root, output_dir='.', c_py_path=None):
    """Orchestrate the full pipeline and write stubs/<name>.pyi."""
    # Step 1-2: scan csrc/ for m.def(...) and parse each TORCH schema.
    m_def_statements = extract_m_def_statements(root)
    parsed_ops = [parse_m_def_statement(stmt) for stmt in m_def_statements]

    # Step 3: read wrapper defaults from deep_gemm/_C.py.
    wrapper_defaults = {}
    if c_py_path is not None:
        c_py_path = Path(c_py_path)
        if c_py_path.is_file():
            wrapper_defaults = parse_c_py_metadata(c_py_path)
        else:
            print(f'Warning: wrapper file not found: {c_py_path}')

    # Steps 4-7: adjust params, apply defaults, render stubs, write file.
    pyi_content = generate_pyi_file_content(
        parsed_ops,
        module_name=name,
        wrapper_defaults=wrapper_defaults,
    )

    output_path = Path(output_dir) / f'{name}.pyi'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pyi_content)

    print(f'.pyi file generated: {output_path}')


def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Generate deep_gemm/_C.pyi stubs from TORCH_LIBRARY schemas.',
    )
    parser.add_argument('--name', default='_C', help='Module name for the .pyi file (default: _C)')
    parser.add_argument('--root', default='./csrc', help='Root to scan for m.def(...) (default: ./csrc)')
    parser.add_argument('--output-dir', default='./stubs', help='Output directory (default: ./stubs)')
    parser.add_argument(
        '--c-py',
        default='./deep_gemm/_C.py',
        help='Python wrapper module to read public API defaults from (default: ./deep_gemm/_C.py)',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Verify the output has typed stubs (no generic *args, **kwargs)',
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    if not root.is_absolute():
        root = repo_root / root
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    c_py_path = Path(args.c_py)
    if not c_py_path.is_absolute():
        c_py_path = repo_root / c_py_path

    generate_pyi_file(
        name=args.name,
        root=str(root),
        output_dir=str(output_dir),
        c_py_path=str(c_py_path),
    )

    pyi_path = output_dir / f'{args.name}.pyi'
    if args.check:
        content = pyi_path.read_text(encoding='utf-8')
        generic_count = content.count('*args, **kwargs')
        stub_count = content.count('def ')
        if stub_count == 0:
            print(f'CHECK FAILED: no function stubs in {pyi_path}', file=sys.stderr)
            return 1
        print(
            f'CHECK PASSED: {stub_count} stubs in {pyi_path} '
            f'({stub_count - generic_count} typed, {generic_count} generic)',
        )
        if generic_count > 0:
            return 1
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(main())
