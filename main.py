import spark.cli.fetch as fetch
import spark.cli.pipeline as pipeline
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
    app.add_typer(pipeline.app, name="pipeline")
    app()


if __name__ == "__main__":
    main()
