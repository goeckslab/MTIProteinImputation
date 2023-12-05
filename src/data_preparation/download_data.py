import requests,sys
from pathlib import Path
import pandas as pd

save_path = Path("data", "bxs")

files = {
    "9_2_1": "https://dataverse.harvard.edu/api/access/datafile/7577973",
    "9_2_2": "https://dataverse.harvard.edu/api/access/datafile/7577976",
    "9_3_1": "https://dataverse.harvard.edu/api/access/datafile/7577979",
    "9_3_2": "https://dataverse.harvard.edu/api/access/datafile/7577978",
    "9_14_1": "https://dataverse.harvard.edu/api/access/datafile/7577974",
    "9_14_2": "https://dataverse.harvard.edu/api/access/datafile/7577977",
    "9_15_1": "https://dataverse.harvard.edu/api/access/datafile/7577972",
    "9_15_2": "https://dataverse.harvard.edu/api/access/datafile/7577975",
}

if __name__ == '__main__':
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for file_name, url in files.items():
        print(f"Downloading biopsy {file_name}...")
        # download data by using the request library
        try:
            r = requests.get(url)
            with open(Path(save_path, f"{file_name}.tsv"), "wb") as f:
                f.write(r.content)

            # load file and convert to csv
            df = pd.read_csv(Path(save_path, f"{file_name}.tsv"), sep="\t")
            df.to_csv(Path(save_path, f"{file_name}.csv"), index=False)

            # remove tsv file
            Path(save_path, f"{file_name}.tsv").unlink()

        except KeyboardInterrupt:
            print("Keyboard interrupt. Stopping...")
            sys.exit(0)

        except BaseException as ex:
            print(ex)
            print(f"Could not download {file_name}")
            continue
