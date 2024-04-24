import ast
import hashlib

from IPython.core.error import UsageError
from IPython.core.magic import Magics, cell_magic, magics_class

# Global variables to store gear versions and code hashes
gear_version_counter = {}
gear_code_hashes = {}


# AST utility for extracting specific method code
def extract_method_code(cell, method_names):
    class GearVisitor(ast.NodeVisitor):
        def __init__(self):
            self.method_code = {}

        def visit_FunctionDef(self, node):
            if node.name in method_names:
                # Convert the method node back to source code
                self.method_code[node.name] = ast.get_source_segment(cell, node)

    tree = ast.parse(cell)
    visitor = GearVisitor()
    visitor.visit(tree)

    # Concatenate the method code
    return "".join(visitor.method_code.values())


@magics_class
class GearMagics(Magics):
    # @cell_magic
    # def version(self, line, cell):
    #     """Saves the versions of all gears attached to the session."""
    #     global gear_classes

    #     # Fetch the user-defined session instance
    #     session = self.shell.user_ns.get("session", None)
    #     if not session:
    #         raise UsageError(
    #             "Please initialize a Session instance with the name 'session' before using %%version."
    #         )

    #     # Get the code for all the gears in gear_classes
    #     # TODO use inspect module
    #     gear_codes = {
    #         gear_name: inspect.getsource(gear_classes[gear_name])
    #         for gear_name in gear_classes
    #     }

    #     # Get the examples in the session
    #     examples = session.examples

    #     # Get the results of the examples
    #     results = asyncio.run(session.run(examples))

    #     # Print the results
    #     print(results)

    @cell_magic
    def gear(self, line, cell):
        global gear_version_counter, gear_code_hashes

        # Extract the class name
        gear_name = cell.split("class")[1].split("(")[0].strip()

        # Check if --root is in the line parameters
        is_root = "--root" in line.split()

        # Extract code blocks for prompt, transform, and __init__ methods
        relevant_code = extract_method_code(cell, ["prompt", "__init__", "editHistory"])

        # Calculate hash of the extracted code
        code_hash = hashlib.md5(relevant_code.encode()).hexdigest()

        # Fetch the user-defined session instance
        session = self.shell.user_ns.get("session", None)
        if not session:
            raise UsageError(
                "Please initialize a Session instance with the name 'session' before using %%gear."
            )

        # Execute the cell (defines the class in the IPython environment)
        self.shell.run_cell(cell)

        # Manage versioning
        gear_class = self.shell.user_ns[gear_name]
        session._gear_code[gear_name] = cell
        if gear_name in gear_code_hashes:
            # If the gear's code changed, increment version counter
            if gear_code_hashes[gear_name] != code_hash:
                gear_version_counter[gear_name] += 1
                gear_class._version = gear_version_counter[
                    gear_name
                ]  # Update the version in the Gear
            gear_code_hashes[gear_name] = code_hash
        else:
            # Initialize the gear's version counter and store its code hash
            gear_version_counter[gear_name] = 1
            gear_class._version = gear_version_counter[gear_name]
            gear_code_hashes[gear_name] = code_hash

        # If the gear is marked as root, register it as the root gear for the session
        if is_root:
            bypass_error = False
            if session.root:
                if session.root.__class__.__name__ != gear_class.__name__:
                    raise UsageError(
                        "Multiple root gears detected. Only one root gear can be set per session. You may want to restart and try again."
                    )

                # If the root gear is the same, update the version
                bypass_error = True

            session.register_root(self.shell.user_ns[gear_name](), bypass_error)


# Load the magic command into the IPython environment
def load_ipython_extension(ipython):
    ipython.register_magics(GearMagics)
