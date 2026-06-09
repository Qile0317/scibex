"""Console script for scibex."""

import cyclopts
from rich.console import Console

from scibex import utils

app = cyclopts.App()
console = Console()


@app.default
def main():
    """Console script for scibex."""
    console.print("Replace this message by putting your code into scibex.cli.main")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
