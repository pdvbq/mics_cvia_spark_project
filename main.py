from spark.settings import Settings
import spark.cli.fetch as fetch
import spark.tools as tools
import logging
import typer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    app = typer.Typer()
    app.add_typer(fetch.app, name="fetch")

    app()


if __name__ == "__main__":
    main()
