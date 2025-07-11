#extracts all the libraries in a file - useful for then pip installing all the libraries u copy from the cell output
import nbformat

# Load the notebook
with open('scan.py') as f:
    nb = nbformat.read(f, as_version=4)

# Extract code cells
code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']

# List to store libraries
libraries = set()

# Iterate through code cells
for cell in code_cells:
    for line in cell.source.split('\n'):
        line = line.strip()
        if line.startswith('import '):
            lib = line.split(' ')[1]
            libraries.add(lib)
        elif line.startswith('from '):
            lib = line.split(' ')[1]
            libraries.add(lib)

# Display the libraries
print("Libraries to be installed:")
for lib in libraries:
    print(lib)
