import inspect
import sys

def recompile_nb_code():
    this_module = sys.modules[APTType]
    module_members = inspect.getmembers(this_module)

    for member_name, member in module_members:
        if hasattr(member, 'recompile') and hasattr(member, 'inspect_llvm'):
            member.recompile()
            print("done")

recompile_nb_code()
