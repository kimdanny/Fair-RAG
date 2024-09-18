import subprocess
import os


def wget_file_to_dir(url, download_path, custom_file_name) -> None:
    try:
        subprocess.run(
            [
                "wget",
                "-P",
                download_path,
                "-O",
                os.path.join(download_path, custom_file_name),
                url,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to download file: {e}")
    except Exception as e:
        print(f"Error: {e}")
