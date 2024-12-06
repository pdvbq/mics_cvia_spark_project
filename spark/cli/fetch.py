import typer
import spark.tools as tools
from spark.settings import settings

app = typer.Typer()


@app.command()
def stream1():
    tools.download_dataset(
        settings.download_cfg.streams["stream1"],
        "stream1",
        settings.download_cfg.data_dir,
        True,
    )


@app.command()
def stream2():
    tools.download_dataset(
        settings.download_cfg.streams["stream2"],
        "stream2",
        settings.download_cfg.data_dir,
        True,
    )
