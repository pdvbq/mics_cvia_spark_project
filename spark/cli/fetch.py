import typer
import spark.tools as tools
from spark.settings import Settings

app = typer.Typer()


@app.command()
def stream1():
    tools.download_dataset(
        Settings.fetch_cfg.stream1_url, "stream1", Settings.fetch_cfg.data_dir, True
    )


@app.command()
def stream2():
    tools.download_dataset(
        Settings.fetch_cfg.stream2_url, "stream2", Settings.fetch_cfg.data_dir, True
    )
