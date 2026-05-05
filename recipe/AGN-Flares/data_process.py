import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

def convert_mjd_to_date(mjd):
    """
    Convert MJD (Modified Julian Date) to YYYY/MM/DD string
    """
    t = Time(mjd, format='mjd')
    return t.strftime('%Y/%m/%d')

def convert_ra_dec(ra_deg, dec_deg):
    """
    Convert RA/DEC from degrees to hh:mm:ss / dd:mm:ss string
    """
    coord = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree)
    ra_str = coord.ra.to_string(unit=u.hour, sep=':', precision=3)
    dec_str = coord.dec.to_string(sep=':', precision=2, alwayssign=True)
    return ra_str, dec_str

def convert_csv(input_file, output_file, info_source_label):
    """
    Convert original AGN CSV to target CSV format
    """
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist!")
        return

    df = pd.read_csv(input_file, sep=None, engine='python')

    df.columns = df.columns.str.strip()

    required_cols = ['AGN_name', 'RA', 'DEC', 'z', 't0']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {input_file}")
    new_df = pd.DataFrame()
    new_df['name'] = df['AGN_name']

    # Convert t0 (MJD) to maxdate and discoverdate
    new_df['maxdate'] = df['t0'].apply(convert_mjd_to_date)
    new_df['discoverdate'] = ""
    ra_dec = df.apply(lambda row: convert_ra_dec(row['RA'], row['DEC']), axis=1)
    new_df['ra'] = [x[0] for x in ra_dec]
    new_df['dec'] = [x[1] for x in ra_dec]
    new_df['redshift'] = df['z']
    new_df['info_source'] = info_source_label

    # Save to CSV
    new_df.to_csv(output_file, index=False)
    print(f"Converted CSV saved to {output_file}")

if __name__ == "__main__":
    convert_csv("data/AGN-Flares/AGNFCC.csv", "data/AGNFCC-GWEyes.csv", 
                info_source_label="AGNFCC")
    convert_csv("data/AGN-Flares/AGNFRC.csv", "data/AGNFRC-GWEyes.csv",
                info_source_label="AGNFRC")