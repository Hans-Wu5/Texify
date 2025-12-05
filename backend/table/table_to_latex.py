# table_to_latex.py

def escape_latex(text: str) -> str:
    """
    Escape characters that break LaTeX tables.
    """
    if not text:
        return ""

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


def table_cells_to_latex(rows):
    """
    rows = [
        ["Name", "Age", "Score"],
        ["Alice", "22", "91"],
        ["Bob", "19", "88"]
    ]

    → returns LaTeX tabular string.
    """

    if not rows:
        return "$\\text{(empty table)}$"

    # Number of columns determined by the longest row
    max_cols = max(len(r) for r in rows)

    # Add vertical lines between columns
    col_spec = "|" + "|".join(["c"] * max_cols) + "|"

    latex = []
    latex.append("\\begin{tabular}{" + col_spec + "}")
    latex.append("\\hline")

    for row in rows:
        padded = row + [""] * (max_cols - len(row))

        escaped = [escape_latex(cell) for cell in padded]
        line = " & ".join(escaped) + r" \\"
        latex.append(line)
        latex.append("\\hline")

    latex.append("\\end{tabular}")

    return "\n".join(latex)

    return "\n".join(latex)