# Minimal example script (verify Image 2 workflow)
from pywinauto.application import Application
import time

# 1. Start or Connect to the target application (e.g., Calculator)
# Use backend="uia" for modern apps, "win32" for older ones.
app = Application(backend="uia").start("calc.exe")
time.sleep(2) # Give it a moment to load

# 2. Perform the 'Inspection' inside the script.
# This will print the hierarchical control tree seen in Image 2.
app.Calculator.print_control_identifiers()

# If successful, you will see the full control hierarchy in your terminal console.