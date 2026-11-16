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
- **Talk to us**: the [#lincc-frameworks-lsdb](https://lsstc.slack.com/archives/C04610PQW9F) channel on the LSST Discovery Alliance Slack;
  all ways to reach the team are on the [contact page](https://docs.lsdb.io/en/latest/contact.html).
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

from prep import (AGN_CUT, MIN_EPOCHS, MILLIQUAS_PARTITION_ROWS, SELECTION_COLUMNS, STAR_CUT, ZTF_MATCH_RADIUS_ARCSEC, as_catalog, catalog_path, clean_lightcurves, features, infrared_excess, label, selection_steps,
                  field, inside_fraction, keep, kept, lightcurve_table, match,
                  equal_pairs, milliquas, open_euclid, open_gaia, open_rubin, open_ztf, pair_sums,
                  plot_catalog, plot_field, plot_how_it_works, rest_frame, structure_function_of,
                  plot_lightcurves, plot_long_lightcurves, plot_selection, plot_survey_sf, nested_lightcurves, LAG_BINS_ZTF, plot_structure_function,
                  plot_tiling, positions, replay, sky_density, star_sequence, structure_functions_of, watch,
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
print(f"{100 * inside_fraction(euclid_area, cone):.1f} per cent of the Euclid deep objects in this picture are inside the {cone.radius_arcsec / 3600:.1f} degree cone we search.")
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
## Crossmatch all three, once

`a.crossmatch(b, radius_arcsec=1)` returns a new lazy catalog; nothing is read until it is computed.
Chaining two of them matches Rubin to Gaia and the result to Euclid, so every source that comes out is
in all three catalogs. Two more calls work on each partition (a pandas table) on the workers:

- `catalog.map_partitions(features)` computes the quantities the selection needs from the three surveys;
- `catalog.map_partitions(clean_lightcurves)` replaces the raw Rubin photometry by a clean lightcurve,
  kept in a nested column.

The result (44 thousand sources with their lightcurves) is written as a new HATS catalog:
`lsdb.from_dataframe` turns the table into a catalog and `to_hats` writes it. Everything after this cell
reads that catalog; the three surveys are not opened again.

This takes about four minutes on eight workers, so it was run ahead of time; the map replays the
recorded run. Set `compute = True` in the first cell to run it again.
"""),
    code("crossmatch", "Crossmatch Rubin, Gaia and Euclid with lightcurves; write a new HATS catalog", """
matched = (match(match(open_rubin(cone, lightcurves=True), gaia), euclid)
           .map_partitions(features)
           .map_partitions(clean_lightcurves))

picture = dict(what="sources in all three catalogs", others={"Gaia DR3 files": gaia, "Euclid DR1 deep files": euclid})
if compute:
    everything, tiles = watch(matched, client, cone, **picture)
    as_catalog(everything).to_hats(catalog_path(cone, "rubin_gaia_euclid"), catalog_name="rubin_gaia_euclid", overwrite=True)
    keep(cone, "tiles", tiles)
else:
    replay(kept(cone, "tiles"), cone, **picture)

matched = lsdb.open_catalog(catalog_path(cone, "rubin_gaia_euclid"))
print(f"{len(matched):,} sources are in all three catalogs, in {len(matched.get_healpix_pixels())} files")
matched
""", question="Which sources are in all three catalogs?", qtype="association"),
    md("section-select", """
## Selecting AGN candidates, one cut at a time

The selection needs no lightcurves, so only the other columns of the new catalog are read (`columns=`),
which takes about a second. Each survey contributes one cut. Gaia: no parallax and no proper motion.
Euclid: a point source (peak surface brightness minus total magnitude is the same small number for all
point sources and larger for galaxies). Rubin and Euclid together: brighter in the near-infrared than a
star of the same optical colour. The cell prints how many sources remain after each cut.
"""),
    code("agn-select", "One cut per survey selects the AGN candidates", """
start = time.time()
sources = lsdb.open_catalog(catalog_path(cone, "rubin_gaia_euclid"), columns=SELECTION_COLUMNS).compute()
sequence = star_sequence(sources)  # the colours of ordinary stars, from sources that Gaia sees move
sources = infrared_excess(sources, sequence).set_index("objectId")
print(f"selection columns of {len(sources):,} sources read in {time.time() - start:.1f} s\\n")

picked = selection_steps(sources)
print(f"\\n{picked.gaia_quasar.sum()} of the {len(picked):,} selected are in Gaia's own list of quasar candidates.")
plot_selection(sources, picked.index)
plt.show()
""", question="Can colour, shape and astrometry select AGN, using all three surveys?", qtype="association"),
    md("section-lsdb", """
## The selection as a new catalog

The same cuts, as text, go to `catalog.query(...)`, which keeps rows of a lazy catalog with pandas query
syntax. Here it keeps the AGN candidates and a set of comparison stars (sources that Gaia sees move and
that lie on the star sequence), both with at least 50 clean epochs. `to_hats` writes the lazy result
directly as a second, smaller catalog; this runs live.
"""),
    code("sample", "Query the catalog for AGN candidates and comparison stars; write them with to_hats", """
start = time.time()
print("AGN candidates: ", AGN_CUT)
print("comparison stars:", STAR_CUT)
selected = (matched
            .map_partitions(infrared_excess, sequence=sequence)
            .query(f"rubin_epochs >= {MIN_EPOCHS} and (({AGN_CUT}) or ({STAR_CUT}))")
            .map_partitions(label))
selected.to_hats(catalog_path(cone), catalog_name="agn_and_stars", overwrite=True)

agn_stars = lsdb.open_catalog(catalog_path(cone))
print(f"selected and written in {time.time() - start:.0f} s")
agn_stars
""", question="How do I keep a selection as a catalog of its own?"),
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

For each pair of epochs the noise its two reported errors predict, err1² + err2², is subtracted from the
squared difference, so the plot shows the variability that the errors do not explain. The notebook then
merges the time bins so that every point is built from about the same number of AGN pairs, and takes the
median over objects in each band. Stars that do not vary stay near zero; AGN rise with time. The dashed
line is a damped random walk (slope 1/2), the usual AGN model.
"""),
    code("structure-function", "AGN differ more the longer you wait; stars do not", """
start = time.time()
sums = equal_pairs(agn_stars.map_partitions(pair_sums).compute())
print(f"{int(sums.pairs.sum()):,} pairs of epochs, binned on the workers  ({time.time() - start:.0f} s)")

labels = {"AGN": f"AGN candidates ({len(agn):,})", "star": f"stars ({len(calm):,})"}
sf = structure_functions_of(sums, labels, remove_noise=True)
plot_structure_function(sf, title="Structure function with the measurement noise removed")
plt.show()
sf["r"][labels["AGN"]].round(3)
""", question="How does the variability of AGN differ from that of stars?", qtype="association"),
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

observed, with_z = f"AGN with a redshift ({len(known):,}), observed time", "the same AGN, rest-frame time"
sf_rest = {band: {labels["star"]: sf[band][labels["star"]],
                  observed: structure_function_of(sums, "AGN", band, remove_noise=True, ids=known.objectId),
                  with_z: structure_function_of(sums_rest, "AGN", band, remove_noise=True)}
           for band in "griz"}
plot_structure_function(sf_rest, title="Structure function, noise removed: observed time and rest-frame time")
plt.show()
sf_rest["r"][with_z].round(3)
""", question="How does the structure function change in the rest frame?", qtype="association"),
    md("section-rubin-ztf", """
### The same AGN in two surveys

ZTF saw the same sources for seven years in g and r, with noisier epochs (errors of 0.1 to 0.2 mag;
these sources are near its detection limit). The ZTF lightcurves came from the crossmatch above. For the
AGN with a redshift that ZTF observed well, the same two functions are applied to them: `rest_frame`
moves the times to the rest frame and `pair_sums` gives the structure function, with the noise of the
reported errors removed in the same way. In both surveys the structure function rises with the time
between epochs, and g lies above r. ZTF reaches rest-frame time differences of years. Its values rest on
the reported ZTF errors being right, since the noise removed is larger than what remains.
"""),
    code("sf-two-surveys", "The rest-frame structure function of the same AGN from Rubin and from ZTF", """
start = time.time()
z = known.set_index("objectId").redshift
per_band = ztf.groupby(["star", "band"]).size().unstack(fill_value=0)
in_ztf = per_band.index[(per_band["ZTF g"] >= 8) & (per_band["ZTF r"] >= 8)]  # at least 8 ZTF epochs in g and in r
both = z.index[z.index.isin(in_ztf)]  # AGN with a redshift that ZTF observed well
ztf_rest = rest_frame(nested_lightcurves(ztf[ztf.star.isin(both)], redshift=z))
sums_ztf = equal_pairs(pair_sums(ztf_rest, bands=("ZTF g", "ZTF r"), lag_bins=LAG_BINS_ZTF), points=8)
print(f"{int(sums_ztf.pairs.sum()):,} pairs of ZTF epochs of {len(both)} AGN with a redshift  ({time.time() - start:.0f} s)")

two = {"Rubin, six months": {b: structure_function_of(sums_rest, "AGN", b, remove_noise=True, ids=both) for b in ("g", "r")},
       "ZTF, seven years": {b: structure_function_of(sums_ztf, "AGN", b, remove_noise=True) for b in ("ZTF g", "ZTF r")}}
plot_survey_sf(two, frame="rest-frame ", title=f"The same {len(both)} AGN with a redshift in two surveys, noise removed")
plt.show()
""", question="Do Rubin and ZTF show the same variability for the same AGN?", qtype="association"),
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
- About 300 have a Milliquas redshift. In rest-frame time their structure function rises with time and is
  larger in bluer bands, in Rubin and in ZTF lightcurves of the same AGN.

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
