"""Write rubin_gaia_quasars.ipynb: the Scialog demo, one question per cell, with Whybook metadata.

usage: python build_notebook.py            # writes the notebook without outputs
       SCIALOG_HEADLESS=1 jupyter nbconvert --to notebook --execute --inplace rubin_gaia_quasars.ipynb
"""
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = "rubin_gaia_quasars.ipynb"


def md(cell_id, text):
    cell = new_markdown_cell(text.strip())
    cell["id"] = cell_id
    return cell


def code(cell_id, title, source, question=None, qtype="descriptive", written_by="agent", branch=None):
    meta = {"title": title, "written_by": written_by}
    if question:
        meta["asked_by"] = "user"
        meta["question"] = {"id": f"asked:{question}", "text": question, "type": qtype}
    if branch:
        meta["branch"] = branch
    cell = new_code_cell(source.strip(), metadata={"whybook": meta})
    cell["id"] = cell_id
    return cell


cells = [
    md("title", """
# Rubin, Gaia and Euclid in one field, with HATS and LSDB

This notebook is a worked example of [LSDB](https://lsdb.io) and the HATS catalog format. It opens three
large catalogs (Rubin Data Preview 2, Gaia DR3 and Euclid DR1, released on 12 November 2026), crossmatches
them in one Euclid deep field, selects AGN candidates, writes them out as a new catalog and analyses
their lightcurves on Dask workers. The science is an example; the steps are the same for any catalogs.

Where to go from here:

- **Catalogs ready to use**: [data.lsdb.io](https://data.lsdb.io) lists public HATS catalogs (Gaia, ZTF,
  Pan-STARRS, DESI, Euclid, WISE and more) with a copy-and-paste line to open each one.
- **LSDB documentation**: [docs.lsdb.io](https://docs.lsdb.io), with [getting started](https://docs.lsdb.io/en/latest/getting-started.html)
  and [tutorials](https://docs.lsdb.io/en/latest/tutorials.html).
- **The format**: [HATS documentation](https://hats.readthedocs.io) and the [IVOA note](https://www.ivoa.net/documents/Notes/HATS/).
- **Make your own catalog**: [hats-import](https://hats-import.readthedocs.io) for large files; `lsdb.from_dataframe` for a table in memory (used below).
- **Lightcurves in a column**: [nested-pandas](https://nested-pandas.readthedocs.io).
- **Code and questions**: [github.com/astronomy-commons/lsdb](https://github.com/astronomy-commons/lsdb).
- **Talk to us**: the [#lincc-frameworks-lsdb](https://lsstc.slack.com/archives/C04610PQW9F) channel on the LSST Discovery Alliance Slack.
"""),
    md("section-hats", """
## HATS: the sky cut into files

A HATS catalog is a directory of Parquet files. Each file holds the rows of one HEALPix tile of the sky.
Where the sky is crowded the tiles are smaller, so every file holds a similar number of rows.
Every HATS catalog uses the same HEALPix grid, whoever made it. It can sit on a local disk, on a web
server or in cloud storage; `lsdb.open_catalog` takes a path or a URL.

Install with `pip install lsdb`. The catalogs here are local copies at SLAC; from a laptop the same code
works with the URLs listed on [data.lsdb.io](https://data.lsdb.io).
"""),
    code("imports", "Tools and the shared prep module", """
import time

import lsdb
import matplotlib.pyplot as plt
import pandas as pd

from prep import (AGN_CUT, MILLIQUAS_PARTITION_ROWS, REST_SF_AT_DAYS, STAR_CUT, ZTF_MATCH_RADIUS_ARCSEC, as_catalog, catalog_path, clean_lightcurves, features,
                  field, fit_carma, fit_many, fit_partition, inside_fraction, keep, kept, lightcurve_table, match,
                  equal_pairs, milliquas, open_euclid, open_gaia, open_rubin, open_ztf, pair_sums, plot_band_sf, plot_carma,
                  plot_catalog, plot_field, plot_how_it_works, rest_frame, structure_function_of,
                  plot_lightcurves, plot_long_lightcurves, plot_rubin_ztf, plot_selection, plot_structure_function,
                  plot_tiling, positions, replay, sf2_at, sky_density, star_sequence, structure_functions_of, watch,
                  workers, ztf_epochs)

compute = False  # False: load what was saved ahead of time and replay the recorded runs. True: run everything.
client = workers(8, memory="14GB")
""", written_by="user"),
    code("load", "Open Rubin DP2, Gaia DR3, Euclid DR1 and ZTF DR24", """
start = time.time()
rubin_sky = open_rubin()
gaia_sky = open_gaia()
euclid_sky = open_euclid(survey="deep")
ztf_sky = open_ztf()

surveys = {"Rubin DP2": rubin_sky, "Gaia DR3": gaia_sky, "Euclid DR1 deep": euclid_sky,
           "ZTF DR24 lightcurves": ztf_sky}
for name, cat in surveys.items():
    print(f"{name:<20}: {cat.hc_structure.catalog_info.total_rows:>13,} rows in {len(cat.get_healpix_pixels()):>5,} files")
print(f"opened in {time.time() - start:.1f} s: only the list of files was read, no rows")

cone = field()
plot_tiling(surveys, cone)
plt.show()
""", question="What data do we have, and how is it stored?", written_by="user"),
    code("how-it-works", "How HATS and LSDB work", """
plot_how_it_works()
plt.show()
""", question="How do HATS and LSDB work?"),
    code("field", "The field is the whole Euclid Deep Field Fornax", """
euclid_area = sky_density(euclid_sky, cone)
plot_field(cone, {"Rubin DP2": rubin_sky, "Gaia DR3": gaia_sky, "Euclid DR1 deep": euclid_sky},
           density=euclid_area, density_label="where Euclid deep has objects")
plt.show()
print(f"{100 * inside_fraction(euclid_area, cone):.1f} per cent of the Euclid deep objects in this picture are inside the red circle.")
print("A file covers its whole tile; the data cover only the shaded part.")

rubin = open_rubin(cone)
gaia = open_gaia(cone)
euclid = open_euclid(cone)
for name, part, whole in (("Rubin", rubin, rubin_sky), ("Gaia", gaia, gaia_sky), ("Euclid deep", euclid, euclid_sky)):
    print(f"{name}: {len(part.get_healpix_pixels())} of {len(whole.get_healpix_pixels()):,} files to read")
""", question="Which files does our field need?"),
    md("why-fast", """
### Why this is fast

- **Only the files that touch the question are read.** The list of tiles says which ones they are before
  any data is opened. `search_filter=` (a cone, a box, a polygon) and `columns=` are given when opening.
- **All catalogs share one grid.** A Rubin tile is matched against the Gaia tile and the Euclid tile that
  cover it and nothing else. No all-sky index is built. Euclid was converted by a different team (IRSA)
  and fits the same grid.
- **Each group of tiles is an independent task.** Eight Dask workers do eight at once, and the same code
  runs on a laptop, a cluster or in the cloud.
- **Parquet is columnar.** Only the columns we ask for are read. Each Rubin object carries its lightcurve
  in a nested column of the same file, so the crossmatch brings the photometry along.
"""),
    md("section-match", """
## Crossmatch all three

`a.crossmatch(b, radius_arcsec=1)` returns a new lazy catalog; nothing is read until `.compute()`.
Chaining two of them matches Rubin to Gaia and the result to Euclid, so every source that comes out is
in all three catalogs. Printing a lazy catalog (last line of the cell) shows its columns and its number
of partitions without reading data. Without the lightcurves the result is small and runs live.
"""),
    code("crossmatch", "Sources that are in Rubin, Gaia and Euclid", """
start = time.time()
three = match(match(rubin, gaia), euclid)
raw = three.compute()

sequence = star_sequence(features(raw))
sources = features(raw, sequence).set_index("objectId")
print(f"{len(sources):,} sources are in all three catalogs  ({time.time() - start:.0f} s)")
three
""", question="Which sources are in all three catalogs?", qtype="association"),
    md("section-select", """
## Selecting AGN candidates

The crossmatch result is a pandas table, so the selection is ordinary pandas. Each survey contributes one
cut. Gaia: no parallax and no proper motion. Euclid: a point source (peak surface brightness minus total
magnitude is the same small number for all point sources and larger for galaxies). Rubin and Euclid
together: brighter in the near-infrared than a star of the same optical colour. The cut is one line of
text, and the same text is given to LSDB's `query` below.
"""),
    code("agn-select", "One cut per survey selects the AGN candidates", """
print("AGN candidates: ", AGN_CUT)
print("comparison stars:", STAR_CUT)
picked = sources.query(AGN_CUT)
plot_selection(sources, picked.index)
plt.show()
print(f"{len(picked):,} of {len(sources):,} sources pass the AGN cut; "
      f"{picked.gaia_quasar.sum()} of them are in Gaia's own list of quasar candidates.")
""", question="Can colour, shape and astrometry select AGN, using all three surveys?", qtype="association"),
    md("section-lsdb", """
## The same selection inside LSDB, with lightcurves

A table that does not fit in memory should not be computed whole. LSDB gives three tools for working on
it in place, and this cell chains them:

- `catalog.map_partitions(f)`: run your own function `f` on every partition (a pandas table) on the workers;
- `catalog.query("...")`: keep rows, with pandas query syntax;
- `catalog.to_hats(path)`: write the result as a new HATS catalog, which anyone can open and crossmatch.

Here the workers match the three catalogs with the Rubin lightcurves (7 million epochs), compute the
selection quantities, keep the AGN candidates and comparison stars, and clean their lightcurves, one
Rubin file at a time. Only the small result comes back. `lsdb.from_dataframe` turns that table into a
catalog and `to_hats` writes it.

This takes about four minutes on eight workers, so it was run ahead of time; the map replays the
recorded run. Set `compute = True` in the first cell to run it again.
"""),
    code("sample", "Crossmatch, select and clean on the workers; write a new HATS catalog", """
with_lightcurves = match(match(open_rubin(cone, lightcurves=True), gaia), euclid)
sample = (with_lightcurves
          .map_partitions(features, sequence=sequence)
          .query(f"({AGN_CUT}) or ({STAR_CUT})")
          .map_partitions(clean_lightcurves))

others = {"Gaia DR3 files": gaia, "Euclid DR1 deep files": euclid}
if compute:
    table, tiles = watch(sample, client, cone, others=others)
    as_catalog(table).to_hats(catalog_path(cone), catalog_name="agn_and_stars", overwrite=True)
    keep(cone, "tiles", tiles)
else:
    replay(kept(cone, "tiles"), cone, others=others)

agn_stars = lsdb.open_catalog(catalog_path(cone))
agn_stars
""", question="Can the selection and the lightcurve cleaning run inside LSDB?"),
    code("agn-lightcurves", "Lightcurves of AGN candidates (top) and comparison stars (bottom)", """
table = agn_stars.compute()
ep = lightcurve_table(table)
table = table.set_index("objectId")
agn = table[table.kind == "AGN"].sort_values("gaia_G")
calm = table[table.kind == "star"].sort_values("gaia_G")
print(f"The new catalog: {len(agn):,} AGN candidates and {len(calm):,} comparison stars with at least 50 clean epochs, "
      f"{len(ep):,} epochs, in {len(agn_stars.get_healpix_pixels())} files.")

plot_lightcurves(ep, table, list(agn.index[100:103]) + list(calm.index[100:103]))
plt.show()
""", question="What do their Rubin lightcurves look like?"),
    md("section-ztf", """
## A fourth catalog: ZTF lightcurves

Adding a survey is one more `open_catalog` and one more `crossmatch`. ZTF DR24 lightcurves are also a
HATS catalog. `n_neighbors=4` keeps up to four matches per source, because ZTF stores one object per
filter. This catalog is read from IRSA's public S3 bucket; the others are local copies, and only the path
differs. ZTF observed this field from 2018 to 2025, including the first months of Rubin; the AGN candidates
are near its limit, so the epochs are noisy.
"""),
    code("ztf", "ZTF lightcurves of the AGN candidates, 2018 to 2025, next to Rubin", """
start = time.time()
seen = match(positions(agn), open_ztf(cone), radius_arcsec=ZTF_MATCH_RADIUS_ARCSEC, neighbors=4).compute()
ztf = ztf_epochs(seen)
per_agn = ztf.groupby("star").size().reindex(agn.index, fill_value=0)
print(f"{(per_agn > 0).sum():,} of {len(agn):,} AGN candidates have a ZTF lightcurve  ({time.time() - start:.0f} s)")
print(f"ZTF epochs per AGN candidate: median {per_agn.median():.0f}, from {per_agn.min()} to {per_agn.max()}")
print(f"ZTF epochs span MJD {ztf.mjd.min():.0f} to {ztf.mjd.max():.0f}; Rubin epochs span MJD {ep.mjd.min():.0f} to {ep.mjd.max():.0f}")

plot_long_lightcurves(ep, ztf, table, per_agn.sort_values(ascending=False).index[:4])
plt.show()
""", question="What did these AGN do before Rubin looked?"),
    md("section-sf", """
## Structure function, computed on the workers

The structure function is the typical brightness difference between two epochs of a source, as a function
of the time between them. It needs every pair of epochs of every object: 90 million pairs here. This is
the pattern for any per-object statistic in LSDB: write a function that takes one partition and returns
a small table (`pair_sums`), and call `agn_stars.map_partitions(pair_sums).compute()`. The nested
lightcurve column is unpacked inside the function with nested-pandas.

The notebook then merges the time bins so that every point is built from about the same number of AGN
pairs, and takes the median over objects in each band. Stars that do not vary give a flat line at the
noise level; AGN rise with time. The dashed line is a damped random walk (slope 1/2), the usual AGN model.
"""),
    code("structure-function", "AGN differ more the longer you wait; stars do not", """
start = time.time()
sums = equal_pairs(agn_stars.map_partitions(pair_sums).compute())
print(f"{int(sums.pairs.sum()):,} pairs of epochs, binned on the workers  ({time.time() - start:.0f} s)")

labels = {"AGN": f"AGN candidates ({len(agn):,})", "star": f"stars ({len(calm):,})"}
sf = structure_functions_of(sums, labels)
plot_structure_function(sf, title="Structure function as measured")
plt.show()
sf["r"][labels["AGN"]].round(3)
""", question="How does the variability of AGN differ from that of stars?", qtype="association"),
    md("section-noise", """
### Removing the measurement noise

For each pair of epochs the noise its two reported errors predict, err1² + err2², is subtracted from the
squared difference. `pair_sums` already returned both sums, so no data is read again.
"""),
    code("structure-function-noise", "The structure function with the measurement noise removed", """
sf_clean = structure_functions_of(sums, labels, remove_noise=True)
plot_structure_function(sf_clean, title="Structure function with the measurement noise removed")
plt.show()

print("Stars, median over the time bins [mag]. Zero means the reported errors explain the scatter.")
pd.DataFrame({band: {"as measured": sf[band][labels["star"]].sf_mag.median(),
                     "noise removed": sf_clean[band][labels["star"]].sf_mag.median()} for band in sf}).round(3)
""", question="What is left when the measurement noise is removed?", qtype="association"),
    md("section-carma", """
## A model fit for single lightcurves

Any Python code can run on the rows of a catalog. Here EzTaoX (JAX) fits a damped random walk to the
g, r, i and z lightcurves of one source at once: one timescale, one amplitude per band and an extra noise
term per band. The six examples are fitted in the notebook; the next sections give the same function to
LSDB. With six months of data the timescale and amplitude are poorly known one by one, so we quote
their combination, the squared structure function the fit implies at a given time difference.
"""),
    code("carma", "Damped-random-walk fits to AGN candidates with the most data", """
per_band = ztf.groupby(["star", "band"]).size().unstack(fill_value=0).reindex(agn.index, fill_value=0)
in_ztf = per_band.index[(per_band["ZTF g"] >= 8) & (per_band["ZTF r"] >= 8)]
top = agn.loc[in_ztf].rubin_epochs.sort_values(ascending=False).index[:100]
print(f"{len(in_ztf)} AGN candidates are well observed by ZTF.")

start = time.time()
fits = [fit_carma(ep, star) for star in top[:6]]
print(f"{len(fits)} fits in {time.time() - start:.0f} s")
plot_carma(fits, table)
plt.show()
examples = pd.DataFrame([f[0] for f in fits]).set_index("star")
for band in "griz":
    examples[f"SF2(100 d) {band}"] = sf2_at(examples, band)
examples[["epochs", "tau_days", "tau_factor"] + [f"SF2(100 d) {band}" for band in "griz"]].round(4)
""", question="What are the CARMA parameters of single AGN?", qtype="descriptive"),
    md("section-redshift", """
## A catalog that is not in HATS: `lsdb.from_dataframe`

Milliquas (the Million Quasars catalog, with redshifts) is one FITS file on a web site. Any table that
pandas can hold becomes a catalog with one call:

    mq = lsdb.from_dataframe(table, ra_column="mq_ra", dec_column="mq_dec")

It is cut into tiles on the same grid, in memory, and from then on it is crossmatched like any other
catalog. For tables too large for memory, [hats-import](https://hats-import.readthedocs.io) does the same
on disk.

The picture shows the crossmatch result as LSDB holds it: a catalog made of tiles, each tile one piece of
work, with the matched objects in them. We keep the AGN candidates that have a Milliquas redshift.
"""),
    code("milliquas", "Milliquas, turned into a catalog on the fly and crossmatched", """
start = time.time()
mq_table = milliquas(cone)
mq = lsdb.from_dataframe(mq_table, ra_column="mq_ra", dec_column="mq_dec", catalog_name="milliquas",
                         partition_rows=MILLIQUAS_PARTITION_ROWS, highest_order=10)
agn_z = match(agn_stars.query("kind == 'AGN'"), mq).query("redshift > 0")
known = agn_z.compute()
print(f"Milliquas: {len(mq_table):,} entries near the field, in {len(mq.get_healpix_pixels())} files built in memory.")
print(f"{len(known):,} of {len(agn):,} AGN candidates are in Milliquas with a redshift "
      f"(median {known.redshift.median():.2f}, from {known.redshift.min():.2f} to {known.redshift.max():.2f})  ({time.time() - start:.0f} s)")

plot_catalog(agn_z, known, cone, others={"Milliquas files (built in memory)": mq},
             title=f"The crossmatch result: {len(agn_z.get_healpix_pixels())} files, {len(known):,} AGN with a redshift")
agn_z
""", question="Which of our AGN candidates have a known redshift?", qtype="association"),
    md("section-rest", """
### Changing a nested column: time in the rest frame

Light from redshift z is stretched in time by 1 + z, so every time difference is divided by 1 + z before
sources are compared. In LSDB this is one more function for `map_partitions` (`rest_frame`): it rewrites
the times inside the nested lightcurves, partition by partition, and the result is again a lazy catalog.
The structure function is then computed with the same `pair_sums` as before.
"""),
    code("rest-frame", "The structure function in rest-frame time", """
rest = agn_z.map_partitions(rest_frame)
sums_rest = equal_pairs(rest.map_partitions(pair_sums).compute())

with_z = f"AGN with a redshift ({len(known):,}), rest-frame time"
sf_rest = {band: {with_z: structure_function_of(sums_rest, "AGN", band, remove_noise=True),
                  "the same, observed time": structure_function_of(sums, "AGN", band, remove_noise=True, ids=known.objectId)}
           for band in "griz"}
plot_structure_function(sf_rest, title="Structure function, noise removed: observed time and rest-frame time")
plt.show()
sf_rest["r"][with_z].round(3)
""", question="How does the structure function change in the rest frame?", qtype="association"),
    md("section-bands", """
### Watching `map_partitions` fill a catalog

The fit for every AGN with a redshift is `rest.map_partitions(fit_partition)`: the function runs on each
partition, on the rest-frame lightcurves, and returns one row of numbers per object. The map shows the
work as it happens: a tile turns green when its task is back, and each fitted AGN appears as a point
coloured by its redshift. To follow the tasks yourself, as `watch` in `prep.py` does, use
`catalog.to_delayed()` (one Dask task per partition) and `as_completed`. This was run ahead of time on
30 workers; the map replays the recorded run.

Each fit implies a structure function in each band; bluer bands lie higher.
"""),
    code("carma-bands", "AGN with a redshift fitted in the rest frame: bluer bands vary more", """
all_fits = rest.map_partitions(fit_partition)
picture = dict(what="fits", headline="map_partitions(fit_partition): a rest-frame fit for every AGN with a redshift",
               tile="catalog file", done="fitted", value_label="redshift")
if compute:
    client = workers(30, memory="4GB")
    carma, fit_tiles = watch(all_fits, client, cone, points=("ra", "dec", "redshift"), **picture)
    keep(cone, "carma_rest", carma)
    keep(cone, "fit_tiles_rest", fit_tiles)
else:
    replay(kept(cone, "fit_tiles_rest"), cone, **picture)
carma = kept(cone, "carma_rest").set_index("objectId")

plot_band_sf(carma, sf_rest, frame="rest-frame ", longest_days=178 / (1 + carma.redshift.median()),
             at_days=REST_SF_AT_DAYS)
plt.show()
print(f"{len(carma):,} fits.  Median SF^2 at {REST_SF_AT_DAYS:.0f} rest-frame days implied by the fits [mag^2]:  "
      + ",  ".join(f"{b} {sf2_at(carma, b, REST_SF_AT_DAYS).median():.4f}" for b in "griz"))
print(f"Median fitted rest-frame timescale (one per source, shared by g, r, i, z): {carma.tau_days.median():.0f} days "
      f"(middle half {carma.tau_days.quantile(0.25):.0f} to {carma.tau_days.quantile(0.75):.0f})")
""", question="Do AGN vary more at bluer rest-frame wavelengths?", qtype="association"),
    md("section-rubin-ztf", """
### Rubin against ZTF

ZTF saw the same sources for six years, with fewer and noisier epochs. The same model is fitted to the
ZTF lightcurves alone, for the 100 AGN candidates with the most Rubin epochs among those well observed
by ZTF. ZTF has g and r, so for this comparison the Rubin lightcurves are fitted again with g and r only:
both timescales then come from the same two bands. These sources are near ZTF's detection limit, with
errors of 0.1 to 0.2 mag per epoch.
"""),
    code("carma-ztf", "The same model fitted to ZTF alone and to Rubin alone, g and r bands", """
if compute:
    keep(cone, "carma_rubin_gr", fit_many(ep, top, client, bands=("g", "r")))
    keep(cone, "carma_ztf", fit_many(ztf, top, client, bands=("ZTF g", "ZTF r")))
carma_gr, carma_ztf = kept(cone, "carma_rubin_gr"), kept(cone, "carma_ztf")

plot_rubin_ztf(carma_gr, carma_ztf)
plt.show()
ztf_r = carma_ztf.rename(columns={"sigma_ZTF r": "sigma_r"})
print(f"Median timescale: Rubin {carma_gr.tau_days.median():.0f} d, ZTF {carma_ztf.tau_days.median():.0f} d.")
print(f"Median SF^2 at 100 days in r: Rubin {sf2_at(carma_gr, 'r').median():.4f}, ZTF {sf2_at(ztf_r, 'r').median():.4f} mag^2.")
""", question="Do Rubin and ZTF give the same variability parameters?", qtype="association"),
    md("summary", """
## What to take away

**HATS and LSDB**

- A HATS catalog is one Parquet file per sky tile, on a grid shared by all catalogs. A question about one
  field reads about a hundred files out of almost 11,000.
- `open_catalog`, `crossmatch`, `query`, `map_partitions`, `to_hats` and `from_dataframe` were all the LSDB
  this notebook needed. Everything is lazy until `.compute()`, and runs partition by partition on Dask.
- Catalogs made by different teams (Rubin, Gaia, Euclid by IRSA, ZTF, a FITS file from a web site) joined
  with one expression each.
- The result of a pipeline can be written as a new HATS catalog and shared.

**The example science**

- One cut per survey selects about a thousand AGN candidates without using variability.
- Their structure function rises with the time between epochs in every band; that of the stars is flat.
- About 300 have a Milliquas redshift. In rest-frame time, a damped random walk fitted with EzTaoX shows
  that bluer bands vary more.

**Try it yourself**

    pip install lsdb

    import lsdb
    gaia = lsdb.open_catalog("https://data.lsdb.io/hats/gaia_dr3",
                             search_filter=lsdb.ConeSearch(ra=53.1, dec=-27.8, radius_arcsec=600))
    gaia.compute()

Catalogs: [data.lsdb.io](https://data.lsdb.io). Documentation: [docs.lsdb.io](https://docs.lsdb.io).
Every choice in this notebook (field size, match radius, cuts, time bins) is a default in `prep.py`.
"""),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nb.metadata["whybook"] = {
    "mode": "wonder", "outcome": "ir_excess", "unit": "objectId",
    "exploration": {"asked": [], "dismissed": []},
}
nbformat.validate(nb)
nbformat.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
