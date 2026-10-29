"""Execute a notebook in-place using nbclient (avoids jupyter CLI entrypoint conflicts)."""
import sys
import traceback
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name='python3')
try:
    client.execute()
except CellExecutionError as e:
    print("CELL EXECUTION ERROR:")
    print(str(e)[:5000])
finally:
    nbformat.write(nb, path)
print("DONE:", path)
