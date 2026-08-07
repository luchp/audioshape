import filecmp
import os
import tempfile
from pathlib import Path

class OpenWriteChecked:
    def __init__(self, fn: Path, mode: str = "w", chmod: int = None):
        self.fn = fn
        self.mode = mode
        self.chmod = chmod
        self.equal = False
        # Create temp file in same directory with same suffix so tools (e.g.
        # matplotlib) can infer the format from the extension.
        fd, tmp = tempfile.mkstemp(suffix=fn.suffix, dir=fn.parent)
        self.fn_tmp = Path(tmp)
        os.close(fd)

    def __enter__(self):
        self.open_file = self.fn_tmp.open(self.mode)
        return self

    def __exit__(self, *args):
        self.open_file.close()
        try:
            equal = filecmp.cmp(self.fn, self.fn_tmp)
        except FileNotFoundError:
            equal = False
        if equal:
            print(f"File {self.fn} untouched", flush=True)
            self.fn_tmp.unlink()
        else:
            self.fn.unlink(missing_ok=True)
            self.fn_tmp.rename(self.fn)
            print(f"File {self.fn} changed!", flush=True)
            if self.chmod:
                self.fn.chmod(self.chmod)
        self.equal = equal
        return None

def save_figure_checked(fig, path, **kwargs):
    """Save figure only if content changed.
    """
    with OpenWriteChecked(path) as f:
        fig.savefig(f.fn_tmp, **kwargs)
    return f.equal

def write_text_checked(path, content):
    """Write text only if content changed."""
    with OpenWriteChecked(path) as f:
        f.open_file.write(content)
    return f.equal