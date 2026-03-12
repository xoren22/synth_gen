"""AST scan: no ast.Try in new module or new feature functions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ast
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

# New module introduced for this feature: must have zero ast.Try
_NEW_MODULES = [
    "pattern_function_io.py",
]

# New or modified feature functions in existing files
_FEATURE_FUNCTIONS = {
    "antenna_pattern.py": [
        "validate_pattern_function_info",
        "evaluate_pattern_function_db",
        "_eval_isotropic",
        "_eval_latent_fourier",
        "_validate_finite_float",
        "_validate_finite_array",
    ],
}


def _find_try_nodes_in_file(filepath):
    with open(filepath, "r") as f:
        source = f.read()
    tree = ast.parse(source, filename=filepath)
    return [node for node in ast.walk(tree) if isinstance(node, ast.Try)]


def _find_try_in_functions(filepath, func_names):
    with open(filepath, "r") as f:
        source = f.read()
    tree = ast.parse(source, filename=filepath)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in func_names:
                for child in ast.walk(node):
                    if isinstance(child, ast.Try):
                        violations.append((node.name, child.lineno))
    return violations


class TestNewModulesNoTry:
    @pytest.mark.parametrize("module", _NEW_MODULES)
    def test_no_try_except(self, module):
        path = os.path.join(_REPO_ROOT, module)
        assert os.path.isfile(path), f"Module {module} not found"
        try_nodes = _find_try_nodes_in_file(path)
        assert len(try_nodes) == 0, (
            f"{module} contains {len(try_nodes)} try/except block(s) "
            f"at line(s): {[n.lineno for n in try_nodes]}"
        )


class TestFeatureFunctionsNoTry:
    @pytest.mark.parametrize("filename,funcs", _FEATURE_FUNCTIONS.items())
    def test_no_try_in_feature_funcs(self, filename, funcs):
        path = os.path.join(_REPO_ROOT, filename)
        assert os.path.isfile(path), f"File {filename} not found"
        violations = _find_try_in_functions(path, funcs)
        assert len(violations) == 0, (
            f"try/except found in feature functions: "
            f"{[(name, line) for name, line in violations]}"
        )
