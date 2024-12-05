from spark.settings import Settings
import spark.cli.fetch as fetch
import spark.cli.generate as generate
import logging
import coloredlogs
import typer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO", logger=logger, fmt="%(levelname)s[%(name)s] %(message)s"
)


def main():
    app = typer.Typer()
    app.add_typer(fetch.app, name="fetch")
    app.add_typer(generate.app, name="generate")
    app()


if __name__ == "__main__":
    main()
