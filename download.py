# this download script is designed to be run from the command line with no arguments.
# it will read parameters from the config.py file and download era5 data as .nc files using those parameters.
# in our testing, we found CDS request time of ~1hr per variable and download time of ~15min per variable when requesting a 30yr period at 6hr timestep

import cdsapi
from pathlib import Path
from config import dataset, varnames, era5_kwargs, era5_fps, api_credentials_check


def download(dataset, vars, kwargs, fps):
    
    api_credentials_check()
    
    c = cdsapi.Client()
    
    for v, fp in zip(vars, fps):
        kwargs["variable"] = v

        print("Downloading " + dataset + " : " + v + " to " + str(fp) + " ... ")

        c.retrieve(
            dataset,
            kwargs,
            fp
        )
    
        if Path(fp).exists:
            print("Download complete! ERA5 data saved to: " + str(fp))
        else:
            print("Expected output not found, download may have failed.")

    return fp


if __name__ == "__main__":
    
    fp = download(dataset, varnames, era5_kwargs, era5_fps)