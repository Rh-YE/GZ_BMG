import os
from astropy.io import fits
import numpy as np
import pandas as pd
imaging = fits.open("/data/public/renhaoye/object_sdss_imaging.fits")
imaging = imaging[1].data
spec = fits.open("/data/public/renhaoye/object_sdss_spectro.fits")
spec = spec[1].data
ra = np.array(imaging["ra"])
dec = np.array(imaging["dec"])
model_flux = np.array(imaging["MODELFLUX"][:,2])

model_flux[model_flux <= 0] = np.nan
model_mag = -2.5 * np.log10(model_flux) + 22.5
petro_flux = np.array(imaging["PETROFLUX"][:,2])
petro_flux[petro_flux <= 0] = np.nan
petro_mag = -2.5 * np.log10(petro_flux) + 22.5
VAGC_SELECT = np.array(imaging["VAGC_SELECT"])
z = np.array(spec["Z"])
# ra.shape, dec.shape, model_flux.shape, petro_flux.shape, VAGC_SELECT.shape, z.shape

vagc = pd.DataFrame(np.array((ra, dec, model_mag, petro_mag, VAGC_SELECT, z)).T, columns=["ra", "dec", "model_mag", "petro_mag", "VAGC_SELECT", "z"])
vagc["VAGC_SELECT"] = vagc["VAGC_SELECT"].astype(int)
flux_redux = vagc[vagc["VAGC_SELECT"] & 4 != 0].query("z<0.5 and z!=0 and petro_mag<17.77")

import astropy.units as u
from astropy.coordinates import SkyCoord
def match(df_1, df_2, pixel, df1_name):
    """
    match two catalog
    :param df_1:
    :param df_2:
    :return:
    """
    sdss = SkyCoord(ra=df_1.ra, dec=df_1.dec, unit=u.degree)
    decals = SkyCoord(ra=df_2.ra, dec=df_2.dec, unit=u.degree)
    idx, d2d, d3d = sdss.match_to_catalog_sky(decals)
    max_sep = pixel * 0.262 * u.arcsec
    distance_idx = d2d < max_sep
    sdss_matches = df_1.iloc[distance_idx]
    matches = idx[distance_idx]
    decal_matches = df_2.iloc[matches]
    test = sdss_matches.loc[:].rename(columns={"ra": "%s" % df1_name[0], "dec": "%s" % df1_name[1]})
    test.insert(0, 'ID', range(len(test)))
    decal_matches.insert(0, 'ID', range(len(decal_matches)))
    new_df = pd.merge(test, decal_matches, how="outer", on=["ID"])
    return new_df.drop("ID", axis=1)
bright_id = pd.read_parquet("/data1/public/BGS/sim_z/south/bright_id.parquet")
bright_id["ra"] = bright_id["ra"].astype(float)
bright_id["dec"] = bright_id["dec"].astype(float)
redshift = match(bright_id, flux_redux, 2, ["in_ra", "in_dec"]).drop(columns=["ra", "dec"]).rename(columns={"in_ra":"ra", "in_dec":"dec"})
redshift.to_parquet("/data1/public/BGS/sim_z/south/redshift.parquet")