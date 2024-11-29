from spark.settings import Settings
import spark.tools as tools
import logging
import typer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(fetch: str = typer.Option(default="", help="Download dataset")):
    if fetch == "stream1":
        tools.download_dataset(
            Settings.stream1_url, "stream1", Settings.data_root_dir, True
        )
    elif fetch == "stream2":
        tools.download_dataset(
            Settings.stream1_url, "stream2", Settings.data_root_dir, True
        )


if __name__ == "__main__":
    typer.run(main)
