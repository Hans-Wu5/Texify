# matrix_to_latex.py

def tokens_to_matrix_latex(cell_rows, bracket="bmatrix"):
    """
    cell_rows = [["1", "2"], ["3", "4"]]
    """
    lines = []
    for row in cell_rows:
        clean_row = [c if c != "" else " " for c in row]
        lines.append(" & ".join(clean_row))

    body = " \\\\\n".join(lines)
    return f"$\n\\begin{{{bracket}}}\n{body}\n\\end{{{bracket}}}\n$"