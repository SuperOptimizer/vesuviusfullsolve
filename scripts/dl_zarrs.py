import wget
import os


SCROLL5 = "/Volumes/vesuvius/dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/0"
DATAROOT = "/Volumes/vesuvius/dl.ash2txt.org/"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HOMEDIR = os.path.join(THIS_DIR, '..','..')

def dl_scroll1a():
    for z in range(0,5250//128):
        for y in range(0,1675//128):
            for x in range(0,2275//128):
                url = f"https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/2/{z}/{y}/{x}"
                out = f"{HOMEDIR}/dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/2/{z}/{y}"
                os.makedirs(out, exist_ok=True)
                if os.path.exists(out + f'/{x}'):
                    print(f"{out} already exists, skipping")
                    continue
                print(url)
                try:
                    ret = wget.download(url,out)
                    print(ret)
                except:
                    print(f"Download failed for {z}/{y}/{x}")



dl_scroll1a()