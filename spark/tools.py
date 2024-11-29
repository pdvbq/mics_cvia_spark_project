import requests
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
import zipfile
import os
import logging


def download_dataset(
    url: str, name: str, data_dir: str, delete_zip: bool = True
) -> bool:
    """
    Downloads and extracts a dataset from a given URL.

    This function downloads a dataset ZIP file from the specified URL, extracts its contents
    into a specified directory, and optionally deletes the ZIP file after extraction.

    Args:
        url (str): The URL of the dataset ZIP file to download.
        name (str): The name to use for the extracted dataset directory and ZIP file.
        data_dir (str): The path to the directory where the dataset should be stored.
        delete_zip (bool, optional): Whether to delete the ZIP file after extraction. Defaults to True.

    Returns:
        bool: True if the dataset is successfully downloaded and extracted, False otherwise.

    Notes:
        - If the dataset ZIP file already exists, it will not be downloaded again.
        - If the dataset directory already exists, it will not be extracted again.
    """
    logger = logger = logging.getLogger(__name__)
    if not os.path.isdir(data_dir):
        logger.info("Creating data directory")
        os.mkdir(data_dir)

    dataset_zip = os.path.join(data_dir, f"{name}.zip")
    dataset_path = os.path.join(data_dir, name)

    if os.path.exists(dataset_path):
        logger.info(f"{dataset_path} already exists. Not doing anything")
        return True

    if not os.path.exists(dataset_zip):
        logger.info(f"Downloading dataset from {url}")
        try:
            __download_url(url, dataset_zip)
        except Exception as e:
            logger.error(f"Failed to download dataset. Make sure the URL is valid. {e}")
            return False

    if not os.path.exists(dataset_path):
        logger.info("Unzipping dataset")
        try:
            __unzip_dataset(dataset_zip, os.path.join(data_dir))
        except zipfile.BadZipfile:
            logger.error(
                "Failed to unzip dataset. Mostlikely due to bad/corrupted file. Have you used the correct URL?"
            )

            return False

    if delete_zip:
        logger.info(f"Deleting {dataset_zip}")
        try:
            os.remove(dataset_zip)
        except Exception as e:
            logger.error(f"Failed to remove {dataset_zip}. {e}")

    return True


def __download_url(url: str, output_path: str) -> None:
    """Downloads content from a given URL

    Args:
        url(str): URL to fetch dataset
        output_path: path to save the dataset
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024

    with open(output_path, "wb") as file:
        with Progress(
            TextColumn("[bold blue]Downloading:[/bold blue]"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(f"[green]{output_path}", total=total_size)

            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                progress.update(task, advance=len(data))


def __unzip_dataset(dataset_path: str, extract_to: str) -> None:
    """Unzips a dataset.

    Args:
        dataset_path(str): path of the zipped dataset
        extract_to(str): where to extract
    """
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        files = zip_ref.namelist()

        with Progress(
            TextColumn("[bold blue]Extracting:[/bold blue]"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(f"[green]{dataset_path}", total=len(files))

            for file in files:
                zip_ref.extract(file, extract_to)
                progress.update(task, advance=1)
