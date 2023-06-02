# this download script is designed to be run from the command line with no arguments.
# it will read parameters from the config.py file and download era5 data as .nc file using those parameters

import cdsapi
from pathlib import Path
from config import dataset, era5_kwargs, era5_fp, api_credentials_check


def download(dataset, kwargs, fp):
    
    api_credentials_check()
    
    c = cdsapi.Client()
    
    c.retrieve(
        dataset,
        kwargs,
        fp
    )
    
    return fp


if __name__ == "__main__":
    
    print("Downloading " + dataset + " to " + str(era5_fp) + " using these parameters: ")
    print(era5_kwargs)
    
    fp = download(dataset, era5_kwargs, era5_fp)
    
    if Path(fp).exists:
        print("Download complete! ERA5 data saved to: " + str(fp))
    else:
        print("Expected output not found, download may have failed.")