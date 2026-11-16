# Scialog 2026 hackathon: Rubin, Gaia and Euclid in one field, with HATS and LSDB

A demo notebook for the Scialog hackathon (Tucson, November 2026). It is a worked example of
[LSDB](https://lsdb.io) and the HATS catalog format, written to be opened in
[Whybook](https://github.com/openteams-ai/whybook) and extended by asking a model questions.

What the notebook does:

1. Opens Rubin DP2, Gaia DR3, Euclid DR1 (deep fields) and ZTF DR24 as HATS catalogs.
2. Crossmatches Rubin, Gaia and Euclid once, with lightcurves, in Euclid Deep Field Fornax (3.2 degree cone),
   and writes the result as a new HATS catalog.
3. Selects AGN candidates from that catalog, one cut per survey (proper motion, shape, colour), with `query`.
4. Measures the structure function of AGN candidates and stars per band.
5. Imports Milliquas with `lsdb.from_dataframe`, crossmatches it to get redshifts, and moves the
   lightcurves to the rest frame with `map_partitions`.
6. Compares the structure function of the same AGN in Rubin and in ZTF.

## Files

| file | what |
|---|---|
| `rubin_gaia_quasars.ipynb` | the notebook, with Whybook question metadata on every cell |
| `prep.py` | helper functions; every analysis choice is a function default |
| `build_notebook.py` | regenerates the notebook from source (`python build_notebook.py`) |

The notebook is committed without outputs, because Rubin DP2 is available to Rubin data rights holders only.

## Data

- All data paths are in one block at the top of `prep.py` (`SITES`). `SITE = "usdf"` reads the local copies at
  the US Data Facility (SLAC). `SITE = "public"` (or the environment variable `SCIALOG_SITE=public`) reads Gaia
  DR3 from [data.lsdb.io](https://data.lsdb.io) and Euclid DR1 from IRSA's public S3 bucket, from anywhere.
- Rubin DP2 needs Rubin data rights; put the path of your copy in the `rubin` entry.
- ZTF DR24 lightcurves are read from IRSA's public S3 bucket.
- Milliquas v8 is downloaded from quasars.org on first use.

## Running

Python 3.13 with `lsdb 0.11`, `hats 0.11`, `nested-pandas 0.7`, `dask`, `s3fs`, `astropy`, `matplotlib`, `scipy`.
Optional: `pip install "whybook[kernel,models]"`.

The slow step (the crossmatch with lightcurves) saves its result under the
`results` entry of `SITES`. Run once with `compute = True` in the first code cell (the crossmatch takes about 4 minutes on 8 workers);
after that `compute = False` loads the saved result and replays the recorded progress map (the whole notebook
then takes about 2 minutes).

## Questions

- LSDB documentation: [docs.lsdb.io](https://docs.lsdb.io)
- Issues: [github.com/astronomy-commons/lsdb](https://github.com/astronomy-commons/lsdb)
- Contact page: [docs.lsdb.io/en/latest/contact.html](https://docs.lsdb.io/en/latest/contact.html)
- Slack: [#lincc-frameworks-lsdb](https://lsstc.slack.com/archives/C04610PQW9F) on the LSST Discovery Alliance Slack
