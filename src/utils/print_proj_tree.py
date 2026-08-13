# Source - https://stackoverflow.com/a
# Posted by PierreGtch
# Retrieved 2025-12-26, License - CC BY-SA 4.0

from pathlib import Path

def print_tree(p: Path, last=True, header=''):
    elbow = "└──"
    pipe = "│  "
    tee = "├──"
    blank = "   "
    print(header + (elbow if last else tee) + p.name)
    if p.is_dir():
        children = list(p.iterdir())
        for i, c in enumerate(children):
            print_tree(c, header=header + (blank if last else pipe), last=i == len(children) - 1)

if __name__ == '__main__':
    print_tree(Path("."))
