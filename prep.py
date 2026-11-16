"""Helpers for the Scialog demo notebook: HATS tiling, Rubin x Gaia crossmatch, quasars from lightcurves.

Every analysis choice is a module constant used as a function default, so Whybook shows it as a chip
on the cell that calls the function.
"""
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

import os  # noqa: E402

import dask  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

if os.environ.get("SCIALOG_HEADLESS"):  # text progress bars when the notebook is executed without a browser
    import tqdm.auto
    import tqdm.std

    tqdm.auto.tqdm = tqdm.std.tqdm
import lsdb  # noqa: E402

dask.config.set({"distributed.dashboard.link": "http://{host}:{port}/status"})
for _name in ("distributed", "numba", "lsdb", "hats"):
    logging.getLogger(_name).setLevel(logging.ERROR)
plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18, "xtick.labelsize": 15,
                     "ytick.labelsize": 15, "legend.fontsize": 14, "figure.dpi": 90, "savefig.bbox": "tight"})

# ---------------------------------------------------------------------------------------------------------
# Where the data are. This is the only block to change to run the notebook somewhere else: add an entry to
# SITES, or pick one with SITE (or the environment variable SCIALOG_SITE). A path can be a local directory,
# an https:// address or an s3:// bucket; LSDB opens all three the same way.
SITES = {
    "usdf": {  # the US Data Facility at SLAC: local copies
        "rubin": "/sdf/data/rubin/shared/lsdb_commissioning/hats/dp2_rc/object_collection",
        "gaia": "/sdf/data/rubin/cosmicdata/lsdb/gaia_dr3",
        "euclid": "/sdf/data/rubin/cosmicdata/lsdb/euclid_dr1/mer_catalogue_{survey}_survey",
        "ztf": "s3://ipac-irsa-ztf/ztf/enhanced/dr24/lc/hats",
        "results": "/sdf/data/rubin/user/ncaplar/scialog-demo",
    },
    "public": {  # anywhere with a network connection; Rubin DP2 needs Rubin data rights and your own path
        "rubin": "path/to/rubin_dp2/object_collection",
        "gaia": "https://data.lsdb.io/hats/gaia_dr3",
        "euclid": "s3://nasa-irsa-euclid-dr1/contributed/dr1/hats/reformatted_catalogs/mer_catalogue_{survey}_survey",
        "ztf": "s3://ipac-irsa-ztf/ztf/enhanced/dr24/lc/hats",
        "results": "scialog-demo-results",
    },
}
SITE = os.environ.get("SCIALOG_SITE", "usdf")
DP2_PATH = SITES[SITE]["rubin"]
GAIA_PATH = SITES[SITE]["gaia"]
EUCLID_PATH = SITES[SITE]["euclid"]  # Euclid DR1 (released 2026-11-12), converted to HATS by IRSA
ZTF_PATH = SITES[SITE]["ztf"]  # ZTF DR24 lightcurves: one row per object and filter, 2018 to 2025; IRSA's bucket
RESULTS_DIR = SITES[SITE]["results"]  # finished results, one directory per field, so slow steps can run ahead of time
S3_PUBLIC = {"anon": True, "client_kwargs": {"endpoint_url": "https://s3.amazonaws.com"}}  # public buckets, also behind a proxy


def _storage(path):
    """Options LSDB needs to read ``path``: anonymous access for a public S3 bucket, nothing otherwise."""
    return S3_PUBLIC if path.startswith("s3://") else None


# ZTF positions are less precise, and each source can have one ZTF object per filter.
ZTF_MATCH_RADIUS_ARCSEC = 1.5
ZTF_FILTERS = {1: "ZTF g", 2: "ZTF r", 3: "ZTF i"}

# The patch of sky: the whole Euclid Deep Field Fornax. Its centre is a Rubin deep-drilling field (ECDFS).
# A radius of 0.3 gives a quick run on the centre only.
FIELD_RA, FIELD_DEC = 52.95, -28.09
FIELD_RADIUS_DEG = 3.2

# Largest Rubin-Gaia separation accepted as the same star.
MATCH_RADIUS_ARCSEC = 1.0

# A forced-photometry epoch is dropped if any of these flags is set.
BAD_FLAGS = ("psfFlux_flag", "pixelFlags_bad", "pixelFlags_saturated", "pixelFlags_edge", "pixelFlags_cr")
# An epoch is used only if its flux is measured at this signal-to-noise or better.
MIN_SNR = 5.0
# An epoch is dropped when it differs from its own night by this many robust standard deviations.
CLIP_SIGMA = 5.0
# No measurable parallax: |parallax / error| below this. A quasar is too far away to show one.
PARALLAX_SNR_MAX = 3.0
# A source "does not move" when its Gaia proper motion is below this many times its error.
MOTION_SNR_MAX = 3.0
# Time differences [days] between two epochs, binned for the structure function. The first bin is "the same night".
# Pairs are first counted in these fine time bins [days] on the workers; ``equal_pairs`` then merges them into
# SF_POINTS bins that hold about the same number of pairs each.
LAG_BINS = (0.01, 0.03, 0.1, 0.5, *np.arange(1.5, 200, 1.0))  # within a night, then one bin per day
SF_POINTS = 10
# Fine time bins for ZTF, whose lightcurves span seven years: one per day.
LAG_BINS_ZTF = tuple(np.arange(0.5, 3000, 1.0))
# A source needs this many clean Rubin epochs to enter the structure function or a CARMA fit.
MIN_EPOCHS = 50
# Bands used for the structure function and the CARMA fits (u and y have few epochs).
SF_BANDS = ("g", "r", "i", "z")
# Comparison stars lie within this distance [mag] of the star sequence ...
ON_SEQUENCE_MAG = 0.1
# ... and Gaia measures their parallax or proper motion at this significance or better.
STAR_MOTION_SNR_MIN = 5.0

GAIA_COLUMNS = ["source_id", "ra", "dec", "parallax", "parallax_over_error", "pmra", "pmra_error", "pmdec", "pmdec_error",
                "phot_g_mean_mag", "bp_rp", "phot_variable_flag", "in_qso_candidates"]
EUCLID_COLUMNS = ["object_id", "flux_y_templfit", "flux_h_templfit", "mumax_minus_mag"]
# Euclid peak surface brightness minus magnitude: stars are near -2.9, galaxies near -0.8.
EXTENDED_MIN = -2.3
# Gaia sources brighter than this saturate in Euclid's infrared images.
EUCLID_BRIGHT_G = 17.5
# A source is "infrared-bright" when its Y - H is this far above stars of the same g - r (their scatter is 0.07).
EXCESS_MIN_MAG = 0.2
LC_FIELDS = ["band", "midpointMjdTai", "psfFlux", "psfFluxErr", *BAD_FLAGS]
# Bluest to reddest band: purple, green, red, brown, black; y (few epochs) in grey.
BAND_COLORS = {"u": "tab:blue", "g": "tab:green", "r": "tab:red", "i": "peru", "z": "k", "y": "0.6"}




def _results_path(cone):
    return os.path.join(RESULTS_DIR, f"field_{cone.ra:.3f}_{cone.dec:.3f}_{cone.radius_arcsec / 3600:.2f}deg")


def catalog_path(cone, name="agn_and_stars"):
    """Where a catalog we write for this field goes: ``rubin_gaia_euclid`` (every matched source) or
    ``agn_and_stars`` (the selected ones)."""
    return os.path.join(_results_path(cone), name)


def keep(cone, name, table):
    """Save a small table next to the catalog of this field, so a slow step can be run ahead of time."""
    os.makedirs(_results_path(cone), exist_ok=True)
    table.to_parquet(os.path.join(_results_path(cone), f"{name}.parquet"))
    return table


def kept(cone, name):
    """Load a table saved with ``keep``."""
    return pd.read_parquet(os.path.join(_results_path(cone), f"{name}.parquet"))


def field(ra=FIELD_RA, dec=FIELD_DEC, radius_deg=FIELD_RADIUS_DEG):
    """The cone of sky the notebook works in."""
    return lsdb.ConeSearch(ra, dec, radius_deg * 3600)


def open_rubin(region=None, lightcurves=False, columns=None):
    """Rubin DP2 objects: position and coadd magnitudes, optionally with the forced-photometry lightcurve."""
    columns = list(columns or ["objectId", "coord_ra", "coord_dec", "g_psfMag", "r_psfMag", "i_psfMag"])
    if lightcurves:
        columns += [f"objectForcedSource.{f}" for f in LC_FIELDS]
    return lsdb.open_catalog(DP2_PATH, columns=columns, search_filter=region, storage_options=_storage(DP2_PATH))


def open_gaia(region=None, columns=None):
    """Gaia DR3: position, parallax, G magnitude, colour, and Gaia's own
    variability and quasar-candidate flags."""
    return lsdb.open_catalog(GAIA_PATH, columns=list(columns or GAIA_COLUMNS), search_filter=region,
                             storage_options=_storage(GAIA_PATH))


def open_euclid(region=None, survey="deep", columns=None):
    """Euclid DR1 objects: the ``wide`` survey, or the ``deep`` fields (our field is one of them).
    Infrared fluxes in the Y and H bands and how concentrated the light is."""
    return lsdb.open_catalog(EUCLID_PATH.format(survey=survey), columns=list(columns or EUCLID_COLUMNS),
                             search_filter=region, storage_options=_storage(EUCLID_PATH))


def match(left, right, radius_arcsec=MATCH_RADIUS_ARCSEC, neighbors=1):
    """The ``neighbors`` nearest objects of ``right`` within ``radius_arcsec`` of each object of ``left``.
    Only objects with a match are kept. Calls can be chained to match three catalogs."""
    out = left.crossmatch(right, radius_arcsec=radius_arcsec, n_neighbors=neighbors, suffix_method="overlapping_columns")
    return out.map_partitions(lambda df: df.drop(columns="_dist_arcsec"))


def open_ztf(region=None):
    """ZTF DR24 lightcurves (g, r, i from Palomar, from 2018), read from IRSA's S3 bucket."""
    return lsdb.open_catalog(ZTF_PATH, columns=["objectid", "filterid", "objra", "objdec", "lightcurve"],
                             search_filter=region, storage_options=_storage(ZTF_PATH))


def positions(sources):
    """A table of sources (indexed by objectId) as a small LSDB catalog, so it can be crossmatched like any other."""
    return lsdb.from_dataframe(sources.reset_index()[["objectId", "ra", "dec"]], ra_column="ra", dec_column="dec")


def ztf_epochs(seen):
    """ZTF lightcurves as one long table, like ``lightcurve_table``: ``star`` (the objectId), ``band``, ``mjd``,
    ``mag``, ``mag_err``. ``seen`` is the crossmatch of ``positions(sources)`` with ZTF. Epochs that ZTF flags
    are dropped."""
    seen = seen.reset_index(drop=True)
    flat = seen["lightcurve"].nest.to_flat()
    row = flat.index.to_numpy()
    out = pd.DataFrame({"star": seen.objectId.to_numpy()[row], "band": seen.filterid.map(ZTF_FILTERS).to_numpy()[row],
                        "mjd": flat["hmjd"].to_numpy(dtype=float), "mag": flat["mag"].to_numpy(dtype=float),
                        "mag_err": flat["magerr"].to_numpy(dtype=float)})
    return out[(flat["catflags"].to_numpy() == 0)].reset_index(drop=True)


def epochs(matched, bad_flags=BAD_FLAGS, min_snr=MIN_SNR, clip_sigma=CLIP_SIGMA):
    """All lightcurves as one long table, one row per clean epoch.

    ``star`` is the row number in ``matched``. ``mag`` is the AB magnitude and ``mag_err`` its reported error.
    ``rel`` is the flux divided by the star's median flux in that band, minus one. Flagged epochs and epochs
    below ``min_snr`` go, and so do epochs that disagree with the rest of their own night by more than
    ``clip_sigma`` times the larger of the star's robust scatter and the epoch's own error, so a noisy epoch
    is not removed for being noisy."""
    flat = matched.reset_index(drop=True)["objectForcedSource"].nest.to_flat()
    out = pd.DataFrame({"star": flat.index.to_numpy(), "band": flat["band"].astype(str).to_numpy()})
    out["mjd"] = flat["midpointMjdTai"].to_numpy(dtype=float, na_value=np.nan)
    out["flux"] = flat["psfFlux"].to_numpy(dtype=float, na_value=np.nan)
    out["err"] = flat["psfFluxErr"].to_numpy(dtype=float, na_value=np.nan)
    flagged = np.zeros(len(out), dtype=bool)
    for k in bad_flags:
        flagged |= flat[k].to_numpy(dtype=bool, na_value=True)
    out = out[np.isfinite(out.flux) & (out.err > 0) & (out.flux > min_snr * out.err) & ~flagged].reset_index(drop=True)
    out["mag"] = 31.4 - 2.5 * np.log10(out.flux)
    out["mag_err"] = 1.0857 * out.err / out.flux
    out["rel"] = out.flux / out.groupby(["star", "band"]).flux.transform("median") - 1.0
    resid = out.rel - out.groupby(["star", np.floor(out.mjd)]).rel.transform("median")
    scatter = 1.4826 * resid.abs().groupby(out.star).transform("median")
    allowed = np.maximum(scatter.clip(lower=0.005), out.err / out.flux)
    return out[resid.abs() <= clip_sigma * allowed].reset_index(drop=True)


def _ab_mag(flux_ujy):
    flux = np.asarray(flux_ujy, dtype=float)
    return 23.9 - 2.5 * np.log10(np.where(flux > 0, flux, np.nan))


ZTF_COLORS = {"ZTF g": "limegreen", "ZTF r": "salmon", "ZTF i": "peru"}


def plot_lightcurve(ax, lc, ztf=None, bands=None):
    """Magnitude minus the median of its band against time, one colour per band.
    ``ztf`` adds ZTF epochs as squares, each ZTF band relative to its own median."""
    for band, color in BAND_COLORS.items():
        m = (lc.band == band).to_numpy()
        if m.sum() and (bands is None or band in bands):
            ax.errorbar(lc.mjd[m] - 60000, lc.mag[m] - lc.mag[m].median(), yerr=lc.mag_err[m], fmt="o", ms=5, lw=1, capsize=2,
                        color=color, label=f"{band} ({m.sum()})")
    for band, color in ZTF_COLORS.items() if ztf is not None else ():
        m = (ztf.band == band).to_numpy()
        if m.sum():
            ax.errorbar(ztf.mjd[m] - 60000, ztf.mag[m] - ztf.mag[m].median(), yerr=ztf.mag_err[m], fmt="s", ms=5, capsize=2,
                        lw=0.8, color=color, mfc="w", label=f"{band} ({m.sum()})")
    ax.invert_yaxis()
    ax.set_xlabel("MJD $-$ 60000")
    ax.set_ylabel("mag $-$ band median")
    ax.legend(ncol=3, frameon=True, framealpha=0.95, edgecolor="0.3", fontsize=11)


def plot_lightcurves(ep, sources, stars, ncols=3):
    """A grid of Rubin lightcurves for the given ``stars`` (index values of ``sources``). All panels share
    one magnitude range, symmetric about zero and set by the 98th percentile of the offsets over all panels. An epoch outside the range is drawn as a small
    arrow at the edge, pointing to where it lies."""
    stars = list(stars)
    shown = ep[ep.star.isin(stars)]
    offset = (shown.mag - shown.groupby(["star", "band"]).mag.transform("median")).abs()
    limit = 1.3 * float(np.percentile(offset, 98))
    nrows = int(np.ceil(len(stars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 4.8 * nrows), squeeze=False)
    for ax, star in zip(axes.ravel(), stars):
        row = sources.loc[star]
        plot_lightcurve(ax, ep[ep.star == star])
        kind = {"AGN": "AGN candidate, ", "star": "star, "}.get(row.get("kind"), "")
        ax.set_title(f"{kind}G = {row.gaia_G:.1f}")
        ax.set_ylim(limit, -limit)
        lc = ep[ep.star == star]
        off = lc.mag - lc.groupby("band").mag.transform("median")
        for band, color in BAND_COLORS.items():  # epochs outside the range: an arrow at the edge, in the band's colour
            for up in (True, False):
                out = (lc.band == band) & ((off < -limit) if up else (off > limit))
                ax.plot(lc.mjd[out] - 60000, np.full(out.sum(), (-0.94 if up else 0.94) * limit), "^" if up else "v",
                        ms=9, color=color, mec="k", mew=0.6, zorder=6)
    for ax in axes.ravel()[len(stars):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_long_lightcurves(ep, ztf, sources, stars, bands=("g", "r")):
    """One row per source. Left: ZTF (squares, 2018 to 2025) and Rubin in the same ``bands``.
    Right: the Rubin part alone, in all bands."""
    stars = list(stars)
    fig, axes = plt.subplots(len(stars), 2, figsize=(22, 4.8 * len(stars)), squeeze=False, width_ratios=[1.4, 1])
    for (left, right), star in zip(axes, stars):
        lc = ep[ep.star == star]
        plot_lightcurve(left, lc, ztf=ztf[ztf.star == star], bands=bands)
        left.set_title(f"G = {sources.gaia_G[star]:.1f}: ZTF and Rubin, {' and '.join(bands)} bands")
        plot_lightcurve(right, lc)
        right.set_title("Rubin alone, all bands")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------
# Pictures of the HATS tiling
# ---------------------------------------------------------------------------------------------------------
ORDER_COLORS = {2: "#440154", 3: "#46327e", 4: "#365c8d", 5: "#277f8e", 6: "#1fa187", 7: "#4ac16d",
                8: "#a0da39", 9: "#fde725"}


def _outlines(pixels, step=3):
    """Outline of each HEALPix tile, in the order given: (order, array of RA, Dec vertices in degrees)."""
    from cdshealpix.nested import vertices

    out = [None] * len(pixels)
    by_order = {}
    for i, p in enumerate(pixels):
        by_order.setdefault(int(p.order), []).append(i)
    for order, rows in by_order.items():
        lon, lat = vertices(np.array([pixels[i].pixel for i in rows], dtype=np.uint64), order, step=step)
        for i, a, b in zip(rows, lon.deg, lat.deg):
            out[i] = (order, np.column_stack([a, b]))
    return out


def _mollweide(lon_deg, lat_deg):
    """Mollweide projection with RA increasing to the left. ``lon_deg`` must already be in -180..180."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    theta = lat.copy()
    for _ in range(12):  # Newton iteration for 2 theta + sin(2 theta) = pi sin(lat)
        theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / np.maximum(2 + 2 * np.cos(2 * theta), 1e-6)
    return -np.asarray(lon_deg, dtype=float) / 90.0 * np.cos(theta), np.sin(theta)


def _sky_xy(poly):
    """Tile outline in RA, Dec degrees to map coordinates. A tile that crosses RA = 180 is returned as
    two pieces, one on each edge of the map."""
    lon = np.where(poly[:, 0] > 180, poly[:, 0] - 360, poly[:, 0])
    if lon.max() - lon.min() <= 180:
        return [np.column_stack(_mollweide(lon, poly[:, 1]))]
    return [np.column_stack(_mollweide(np.where(lon < 0, 180.0, lon), poly[:, 1])),
            np.column_stack(_mollweide(np.where(lon > 0, -180.0, lon), poly[:, 1]))]


def _sky_axes(ax, title):
    """Empty all-sky map: outline, a coordinate grid every 30 degrees, and declination labels."""
    line = np.linspace(-90, 90, 91)
    for lon in range(-180, 181, 60):
        ax.plot(*_mollweide(np.full_like(line, lon), line), color="k" if abs(lon) == 180 else "0.75",
                lw=1.5 if abs(lon) == 180 else 0.6, zorder=3)
    for lat in range(-60, 61, 30):
        ax.plot(*_mollweide(2 * line, np.full_like(line, lat)), color="0.75", lw=0.6, zorder=3)
        x, y = _mollweide(180, lat)
        ax.text(x - 0.04, y, f"{lat}$^\\circ$", ha="right", va="center", fontsize=13, clip_on=True)
    ax.set_title(title, pad=12)
    ax.set_xlim(-2.25, 2.05)
    ax.set_ylim(-1.03, 1.03)
    ax.set_aspect("equal")
    ax.axis("off")


def _mark_field(ax, cone, label):
    import matplotlib.patheffects as pe

    x, y = _mollweide(cone.ra - 360 if cone.ra > 180 else cone.ra, cone.dec)
    halo = [pe.withStroke(linewidth=6, foreground="w")]
    ax.annotate(label, (x, y), xytext=(x + 0.25, y - 0.5), color="red", fontsize=17, fontweight="bold",
                ha="center", zorder=6, bbox=dict(boxstyle="round,pad=0.3", facecolor="w", edgecolor="red", lw=2),
                arrowprops=dict(arrowstyle="->", color="red", lw=3, path_effects=halo))
    ax.plot(x, y, marker="o", ms=13, mfc="none", mec="red", mew=2.5, zorder=6, path_effects=halo)


def plot_tiling(catalogs, cone=None, label="our field", ncols=2):
    """All-sky map of the files of each catalog. One polygon is one Parquet file; the colour is the tile
    size, so small tiles sit where the sky is crowded and every file holds a similar number of rows."""
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    nrows = -(-len(catalogs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 6.0 * nrows + 0.6), squeeze=False)
    for ax in axes.flat[len(catalogs):]:
        ax.set_axis_off()
    orders = set()
    for ax, (name, cat) in zip(axes.flat, catalogs.items()):
        tiles = _outlines(cat.get_healpix_pixels())
        orders |= {o for o, _ in tiles}
        polys, colors, edges = [], [], []
        for o, p in tiles:
            for piece in _sky_xy(p):
                polys.append(piece)
                colors.append(ORDER_COLORS[o])
                edges.append("w" if o <= 4 else ORDER_COLORS[o])
        ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=edges, linewidths=0.3))
        rows = cat.hc_structure.catalog_info.total_rows
        _sky_axes(ax, f"{name}: {rows / 1e6:,.0f} million rows in {len(tiles):,} files")
        if cone is not None:
            _mark_field(ax, cone, label)
    fig.legend(handles=[Patch(color=ORDER_COLORS[o], label=f"{58.6 / 2 ** o:.1f}$^\\circ$") for o in sorted(orders)],
               title="tile size", loc="lower center", ncol=len(orders), frameon=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.5 / fig.get_figheight(), bottom=0.9 / fig.get_figheight(),
                        wspace=0.02, hspace=0.12)
    return fig


def sky_density(catalog, cone, width_deg=9.0, pixel_deg=0.05):
    """Where a catalog has objects: counts in ``pixel_deg`` pixels of RA and Dec around the cone.
    Only the two position columns are read. Returns the counts and the RA and Dec pixel edges."""
    info = catalog.hc_structure.catalog_info
    ra_col, dec_col = info.ra_column, info.dec_column
    scale = np.cos(np.radians(cone.dec))
    ra_edges = np.arange(cone.ra - width_deg / 2 / scale, cone.ra + width_deg / 2 / scale + pixel_deg, pixel_deg / scale)
    dec_edges = np.arange(cone.dec - width_deg / 2, cone.dec + width_deg / 2 + pixel_deg, pixel_deg)

    def count(df):
        h = np.histogram2d(df[ra_col].to_numpy(dtype=float), df[dec_col].to_numpy(dtype=float), bins=[ra_edges, dec_edges])[0]
        return pd.DataFrame({"counts": [h.astype(np.int32)]})

    wide = lsdb.ConeSearch(cone.ra, cone.dec, width_deg * 3600)
    parts = catalog.search(wide)[[ra_col, dec_col]].map_partitions(count, meta={"counts": object}).compute()
    return np.sum(list(parts.counts), axis=0), ra_edges, dec_edges


def inside_fraction(density, cone):
    """Fraction of the objects counted by ``sky_density`` that lie inside the cone."""
    counts, ra_edges, dec_edges = density
    ra, dec = np.meshgrid(0.5 * (ra_edges[1:] + ra_edges[:-1]), 0.5 * (dec_edges[1:] + dec_edges[:-1]), indexing="ij")
    sep = np.hypot((ra - cone.ra) * np.cos(np.radians(cone.dec)), dec - cone.dec)
    return counts[sep < cone.radius_arcsec / 3600].sum() / max(counts.sum(), 1)


def plot_field(cone, catalogs, width_deg=9.0, density=None, density_label="where the catalog has objects"):
    """Zoom on the cone: the tiles of each catalog around it, and the ones the cone touches (filled).
    Only the filled files are read. ``catalogs`` maps a name to the whole catalog. ``density`` (from
    ``sky_density``) shades the area where a catalog really has objects: a file covers its whole tile,
    the data need not."""
    from matplotlib.patches import Patch, Polygon

    wide = lsdb.ConeSearch(cone.ra, cone.dec, width_deg * 3600)
    styles = {0: dict(edge="tab:blue", fill="tab:blue", lw=1.2, alpha=0.30),
              1: dict(edge="darkorange", fill="orange", lw=4, alpha=0.0),
              2: dict(edge="tab:green", fill="tab:green", lw=2.5, alpha=0.0, ls="--")}
    fig, ax = plt.subplots(figsize=(11, 9.5))
    if density is not None:
        counts, ra_edges, dec_edges = density
        ax.pcolormesh(ra_edges, dec_edges, np.where(counts.T > 0, 1.0, np.nan), cmap="Greens", vmin=0, vmax=2.2, zorder=0)
        ax.add_artist(ax.legend(handles=[Patch(color=plt.cm.Greens(0.45), label=density_label)], loc="lower right",
                                frameon=True, framealpha=0.95, edgecolor="0.3"))
    for k, (name, cat) in enumerate(catalogs.items()):
        style = styles[k]
        used = {(p.order, p.pixel) for p in cat.search(cone).get_healpix_pixels()}
        total = len(cat.get_healpix_pixels())
        for p in cat.search(wide).get_healpix_pixels():
            poly = _outlines([p], step=6)[0][1]
            if (p.order, p.pixel) in used and style["alpha"]:
                ax.add_patch(Polygon(poly, closed=True, facecolor=style["fill"], alpha=style["alpha"], edgecolor="none"))
            ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=style["edge"], lw=style["lw"],
                                 ls=style.get("ls", "-")))
        ax.plot([], [], color=style["edge"], lw=style["lw"], ls=style.get("ls", "-"), label=f"{name} files: {len(used)} of {total:,} read")
    scale = np.cos(np.radians(cone.dec))
    half = width_deg / 2
    ax.set_xlim(cone.ra + half / scale, cone.ra - half / scale)
    ax.set_ylim(cone.dec - half, cone.dec + half)
    ax.set_aspect(1 / scale)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.3")
    return fig


def _quiet():
    warnings.filterwarnings("ignore")
    for name in ("numba", "lsdb", "hats"):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger("distributed").setLevel(logging.CRITICAL)  # no tracebacks when workers are stopped


def workers(n, memory="16GB"):
    """Start ``n`` Dask workers on this machine, replacing any that are already running."""
    from dask.distributed import Client, get_client

    try:
        old = get_client()
        cluster = old.cluster
        old.close()
        if cluster is not None:
            cluster.close()
    except ValueError:
        pass
    client = Client(n_workers=n, threads_per_worker=1, memory_limit=memory, silence_logs=logging.ERROR)
    client.run(_quiet)
    return client






def star_sequence(sources, bin_mag=0.1, min_stars=5, motion_snr_min=STAR_MOTION_SNR_MIN):
    """Median Euclid Y - H of stars as a function of Rubin g - r. Stars are point-like sources whose parallax
    or proper motion Gaia measures at ``motion_snr_min`` or better, so quasars cannot be among them.
    Returns the g - r bin centres and the median Y - H in each."""
    moves = (sources.parallax_snr.fillna(0) > motion_snr_min) | (sources.motion_snr.fillna(0) > motion_snr_min)
    stars = sources[moves & ~sources.euclid_extended & sources.euclid_Y_H.notna() & sources.rubin_g_r.between(-0.5, 2.0)]
    bins = np.floor(stars.rubin_g_r / bin_mag).astype(int)
    med = stars.groupby(bins).euclid_Y_H.agg(["median", "size"])
    med = med[med["size"] >= min_stars]
    return (med.index.to_numpy() + 0.5) * bin_mag, med["median"].to_numpy()


def plot_selection(sources, selected, excess_min_mag=EXCESS_MIN_MAG, extended_min=EXTENDED_MIN,
                   parallax_snr_max=PARALLAX_SNR_MAX, motion_snr_max=MOTION_SNR_MAX):
    """The three cuts of ``select_agn``, one panel per survey. ``selected`` is the index of the chosen sources."""
    s = sources[sources.euclid_Y_H.notna()]
    pick = s.loc[s.index.intersection(selected)]
    sequence = star_sequence(sources)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    grey = dict(s=10, color="0.55")
    red = dict(s=110, color="crimson", edgecolor="k", zorder=4)

    ax = axes[0]
    ax.scatter(s.parallax_snr, np.log10(s.motion_snr), **grey)
    ax.scatter(pick.parallax_snr.fillna(0), np.log10(pick.motion_snr.fillna(0.2)), **red)
    ax.axhline(np.log10(motion_snr_max), color="k", ls="--", lw=2)
    ax.axvline(parallax_snr_max, color="k", ls="--", lw=2)
    ax.set_xlim(-6, 40)
    ax.set_ylim(-1.5, 3.3)
    ax.set_xlabel("parallax / error")
    ax.set_ylabel("log$_{10}$ (proper motion / error)")
    ax.set_title("Gaia: does not move")

    ax = axes[1]
    ax.scatter(s.gaia_G, s.euclid_compactness, **grey)
    ax.scatter(pick.gaia_G, pick.euclid_compactness, **red)
    ax.axhline(extended_min, color="k", ls="--", lw=2)
    box = dict(boxstyle="round", facecolor="w", edgecolor="0.3")
    ax.text(0.03, 0.95, "light spread out: galaxies", transform=ax.transAxes, va="top", fontsize=15, bbox=box)
    ax.text(0.03, 0.05, "all light in one sharp peak: point sources", transform=ax.transAxes, fontsize=15, bbox=box)
    ax.set_ylim(-4.2, 3.5)
    ax.set_xlabel("Gaia G")
    ax.set_ylabel("Euclid: peak surface brightness $-$ total magnitude")
    ax.set_title("Euclid: how concentrated the light is")

    ax = axes[2]
    x, y = star_sequence(sources)
    x, y = np.r_[-0.4, x], np.r_[y[0], y]
    ax.scatter(s.rubin_g_r, s.euclid_Y_H, **grey)
    ax.scatter(pick.rubin_g_r, pick.euclid_Y_H, label=f"selected by all three ({len(pick)})", **red)
    ax.plot(x, y, color="k", lw=2.5)
    ax.plot(x, y + excess_min_mag, color="k", lw=2, ls="--")
    ax.set_xlim(-0.4, 1.8)
    ax.set_ylim(-0.9, 1.3)
    ax.set_xlabel("Rubin g $-$ r")
    ax.set_ylabel("Euclid Y $-$ H")
    ax.set_title("Rubin + Euclid: infrared-bright for its colour")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.3")
    fig.tight_layout()
    return fig



class _TileMap:
    """The picture of the field's tiles used by ``watch`` and ``replay``: grey while waiting, green when done.
    ``others`` (name -> catalog) draws the files of the other catalogs for context."""

    def __init__(self, pixels, cone=None, what="matches", others=None, headline="", tile="Rubin file",
                 done="matched to Gaia and Euclid", value_label=None, value_range=(0.0, 3.5)):
        from matplotlib.patches import Circle, Patch, Polygon

        self.value_range = value_range

        self.what, self.n, self.headline = what, len(pixels), headline
        self.outlines = [poly for _, poly in _outlines(pixels, step=6)]
        self.fig, self.ax = plt.subplots(figsize=(11, 9.5))
        self.patches = [self.ax.add_patch(Polygon(poly, closed=True, facecolor="0.82", edgecolor="tab:blue", lw=1.0))
                        for poly in self.outlines]
        self.tile = tile
        handles = [Patch(facecolor="0.82", edgecolor="tab:blue", label=f"{tile}, waiting"),
                   Patch(facecolor="tab:green", alpha=0.55, edgecolor="tab:blue", label=f"{tile}, {done}")]
        looks = [dict(edgecolor="darkorange", lw=4), dict(edgecolor="tab:green", lw=2.5, ls="--")]
        for look, (name, cat) in zip(looks, (others or {}).items()):
            for _, poly in _outlines(cat.get_healpix_pixels(), step=6):
                self.ax.add_patch(Polygon(poly, closed=True, fill=False, zorder=3, **look))
            handles.append(self.ax.plot([], [], color=look["edgecolor"], lw=look["lw"], ls=look.get("ls", "-"), label=name)[0])
        every = np.vstack(self.outlines)
        scale = np.cos(np.radians(float(np.mean(every[:, 1]))))
        if cone is not None:
            self.ax.add_patch(Circle((cone.ra, cone.dec), cone.radius_arcsec / 3600, fill=False, edgecolor="red",
                                     lw=3, zorder=4))
            handles.append(self.ax.plot([], [], color="red", lw=3, label="the field")[0])
        if cone is not None:  # the picture always holds the whole field
            r = cone.radius_arcsec / 3600
            every = np.vstack([every, [[cone.ra - r / scale, cone.dec - r], [cone.ra + r / scale, cone.dec + r]]])
        self.ax.set_xlim(every[:, 0].max() + 0.1 / scale, every[:, 0].min() - 0.1 / scale)
        self.ax.set_ylim(every[:, 1].min() - 0.1, every[:, 1].max() + 1.6)
        self.ax.set_aspect(1 / scale)
        self.ax.set_xlabel("RA [deg]")
        self.ax.set_ylabel("Dec [deg]")
        if value_label:  # objects are drawn as points, coloured by a value
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            handles.append(self.ax.plot([], [], "o", ms=9, color="tab:purple", mec="k", label="one object")[0])
            self.fig.colorbar(ScalarMappable(Normalize(*value_range), "plasma"), ax=self.ax, label=value_label,
                              fraction=0.04, pad=0.02)
        self.ax.legend(handles=handles, loc="upper left", ncol=2, frameon=True, framealpha=0.95, edgecolor="0.3", fontsize=12)
        plt.close(self.fig)

    def finish(self, i, rows, points=None):
        self.patches[i].set_facecolor("tab:green")
        self.patches[i].set_alpha(0.3 if points is not None else 0.55)
        if points is not None and len(points[0]):
            self.ax.scatter(points[0], points[1], c=points[2], cmap="plasma", vmin=self.value_range[0],
                            vmax=self.value_range[1], s=45, edgecolor="k", lw=0.5, zorder=5)
        if self.n <= 12:
            self.ax.text(*self.outlines[i].mean(axis=0), f"{rows:,}", ha="center", va="center", fontsize=15,
                         fontweight="bold")

    def show(self, done, rows, seconds, note=""):  # clear-and-redraw: the Whybook view follows this live
        import io

        from IPython.display import Image, clear_output, display

        self.ax.set_title(f"{self.headline}\n{done} of {self.n} {self.tile}s done    {rows:,} {self.what}    "
                          f"{seconds:.0f} s{note}", pad=12, fontsize=16)
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=80)
        clear_output(wait=True)
        display(Image(data=buf.getvalue()))


HEADLINE = "Rubin + lightcurves, matched to Gaia and Euclid, lightcurves cleaned: one Rubin file at a time"


def watch(catalog, client, cone=None, what="AGN candidates and stars", refresh_s=1.0, others=None, headline=HEADLINE,
          tile="Rubin file", done="matched to Gaia and Euclid", points=None, value_label=None):
    """Compute ``catalog`` one tile at a time and draw the field as the tiles finish.

    Each tile is one independent task: read one Rubin file and the files of the other catalogs that cover
    it, and match them. A tile is grey while it waits and green when it is done (with its number of rows, if
    there are few tiles). ``others`` (name -> catalog) draws the files of the other catalogs. ``points``
    (names of three columns: RA, Dec and a value) also draws every returned object, coloured by the value. The picture
    is redrawn at most every ``refresh_s`` seconds. Returns all rows as one table, and a record of the run
    (one row per tile: when it finished and how many rows it gave) that ``replay`` can show again."""
    import time

    from dask.distributed import as_completed

    pixels = catalog.get_healpix_pixels()
    picture = _TileMap(pixels, cone, what, others, headline, tile, done, value_label)
    start = last = time.time()
    picture.show(0, 0, 0)
    futures = client.compute(catalog.to_delayed())
    index = {f.key: i for i, f in enumerate(futures)}
    parts, record = [None] * len(pixels), []
    for done, (future, df) in enumerate(as_completed(futures, with_results=True), 1):
        i = index[future.key]
        parts[i] = df
        xyv = [df[c].to_numpy(dtype=float) for c in points] if points else None
        record.append((int(pixels[i].order), int(pixels[i].pixel), time.time() - start, len(df),
                       *([list(v) for v in xyv] if points else [])))
        picture.finish(i, len(df), xyv)
        if time.time() - last > refresh_s or done == len(pixels):
            picture.show(done, sum(r[3] for r in record), time.time() - start)
            last = time.time()
    return pd.concat(parts), pd.DataFrame(record, columns=["order", "pixel", "seconds", "rows",
                                                           *(["x", "y", "value"] if points else [])])


def replay(tiles, cone=None, what="AGN candidates and stars", speedup=10.0, refresh_s=0.5, others=None,
           headline=HEADLINE, tile="Rubin file", done="matched to Gaia and Euclid", value_label=None):
    """Show a recorded run of ``watch`` again, ``speedup`` times faster than it happened.
    ``tiles`` is the record that ``watch`` returned."""
    import time

    from hats.pixel_math import HealpixPixel

    picture = _TileMap([HealpixPixel(o, p) for o, p in zip(tiles.order, tiles.pixel)], cone, what, others, headline,
                       tile, done, value_label)
    note = f"    (recorded run, {speedup:g}x faster)"
    picture.show(0, 0, 0, note)
    start = last = time.time()
    rows = 0
    for done, (i, tile) in enumerate(tiles.reset_index(drop=True).iterrows(), 1):
        wait = tile.seconds / speedup - (time.time() - start)
        if wait > 0:
            time.sleep(wait)
        rows += int(tile.rows)
        picture.finish(i, int(tile.rows), (tile.x, tile.y, tile.value) if "x" in tiles.columns else None)
        if time.time() - last > refresh_s or done == len(tiles):
            picture.show(done, rows, tile.seconds, note)
            last = time.time()


def drw_sf(lag_days, tau_days, sf_inf, noise=0.0):
    """Structure function of a damped random walk: it rises as the square root of the time difference
    (slope 1/2 in a log-log plot) and flattens at ``sf_inf`` beyond ``tau_days``. ``noise`` adds a constant
    measurement-noise floor in quadrature."""
    return np.sqrt(noise ** 2 + sf_inf ** 2 * (1.0 - np.exp(-np.asarray(lag_days, dtype=float) / tau_days)))


def _log_ticks(ax, x=True, y=True):
    from matplotlib.ticker import MultipleLocator

    if x:
        ax.xaxis.set_major_locator(MultipleLocator(1))
    if y:
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(color="0.88", lw=0.7)


def plot_structure_function(panels, title=None):
    """One panel per band (``panels``: band -> {label: structure-function table}). Both axes show log10 values.
    The structure function SF is in magnitudes: the square root of the mean squared difference. The colour
    follows the label: stars grey, AGN red, and AGN in rest-frame time (label contains "rest-frame") purple.
    The dashed line is a damped random walk plus a constant floor (noise that the errors do not account for),
    fitted to the last AGN curve of the panel."""
    from scipy.optimize import curve_fit

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 7), squeeze=False, sharey=True)
    for ax, (band, curves) in zip(axes[0], panels.items()):
        last_agn = [label for label in curves if "AGN" in label][-1:]
        for label, sf in curves.items():
            agn = "AGN" in label
            color = "darkviolet" if "rest-frame" in label else "crimson" if agn else "0.4"
            sf = sf[sf.sf_mag > 0]
            ax.errorbar(np.log10(sf.lag_days), np.log10(sf.sf_mag), yerr=sf.err / sf.sf_mag / np.log(10), fmt="o-", ms=8,
                        lw=2.5, capsize=4, color=color, label=label)
            if [label] == last_agn and len(sf) > 3:
                start = [100.0, float(sf.sf_mag.max()), float(sf.sf_mag.min())]
                try:
                    best, _ = curve_fit(drw_sf, sf.lag_days, sf.sf_mag, p0=start, sigma=sf.err.clip(lower=0.002),
                                        bounds=(0, [3000.0, 2.0, 1.0]))
                    grid = np.geomspace(sf.lag_days.min(), sf.lag_days.max(), 100)
                    ax.plot(np.log10(grid), np.log10(drw_sf(grid, *best)), color="k", ls="--", lw=2.5,
                            label=f"damped random walk (slope 1/2)\n+ floor; timescale {best[0]:.0f} d")
                except (RuntimeError, ValueError):
                    pass
        _log_ticks(ax)
        ax.set_xlabel("log$_{10}$ time between two epochs [days]")
        ax.set_title(f"{band} band")
        ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.3", fontsize=12)
    axes[0][0].set_ylabel("log$_{10}$ structure function SF [mag]")
    axes[0][0].set_ylim(-2.3, -0.3)
    if title:
        fig.suptitle(title, fontsize=19)
    fig.tight_layout()
    return fig


def plot_survey_sf(surveys, title=None, frame=""):
    """Structure functions of several surveys in one panel (``surveys``: survey name -> {band: structure-function
    table}). The colour is the band; the first survey has filled circles and solid lines, the second open
    squares and dashed lines. Both axes show log10 values."""
    styles = [dict(fmt="o-", mec="w"), dict(fmt="s--", mfc="w", mew=2)]
    fig, ax = plt.subplots(figsize=(12, 8))
    for style, (survey, curves) in zip(styles, surveys.items()):
        for band, sf in curves.items():
            sf = sf[sf.sf_mag > 0]
            ax.errorbar(np.log10(sf.lag_days), np.log10(sf.sf_mag), yerr=sf.err / sf.sf_mag / np.log(10), ms=9, lw=2.5,
                        capsize=4, color=BAND_COLORS[band.split()[-1]], label=f"{survey}: {band.split()[-1]} band", **style)
    _log_ticks(ax)
    ax.set_xlabel(f"log$_{{10}}$ {frame}time between two epochs [days]")
    ax.set_ylabel("log$_{10}$ structure function SF [mag]")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.3")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------
# The same analysis inside LSDB: per-partition functions that run on the workers
# ---------------------------------------------------------------------------------------------------------
# The cuts, as text that LSDB's ``query`` understands.
# The AGN selection, one cut per survey. Each entry: what the cut is, and the cut as text for ``query``.
AGN_STEPS = [("Gaia: no parallax and no proper motion", "no_parallax and no_motion"),
             ("Euclid: a point source", "not euclid_extended"),
             ("Rubin and Euclid: near-infrared excess over stars", "ir_excess > 0.2")]
AGN_CUT = " and ".join(cut for _, cut in AGN_STEPS)
STAR_CUT = "moves and on_star_sequence and not euclid_extended and not gaia_quasar"
# Objects per file of the catalog we write.
SAMPLE_PARTITION_ROWS = 300
FEATURES = ["objectId", "ra", "dec", "gaia_G", "parallax_snr", "no_parallax", "motion_snr", "no_motion", "moves",
            "gaia_quasar", "rubin_g_r", "euclid_Y_H", "euclid_compactness", "euclid_extended"]
# The columns of the matched catalog that the selection reads: everything except the lightcurves.
SELECTION_COLUMNS = FEATURES + ["rubin_epochs"]


def features(df, parallax_snr_max=PARALLAX_SNR_MAX, motion_snr_max=MOTION_SNR_MAX,
             motion_snr_min=STAR_MOTION_SNR_MIN, extended_min=EXTENDED_MIN, bright_g=EUCLID_BRIGHT_G):
    """Runs on one partition of the three-way match: adds the quantities the selection uses.

    Gaia: G magnitude, parallax and proper motion divided by their errors; ``no_parallax`` and ``no_motion``
    mark values below ``parallax_snr_max`` and ``motion_snr_max``, ``moves`` marks either above ``motion_snr_min``.
    Rubin: g - r. Euclid: Y - H (empty for sources brighter than ``bright_g``, which saturate) and
    ``euclid_extended`` for sources whose light is spread out (``mumax_minus_mag`` above ``extended_min``)."""
    out = df.copy()
    snr = df.parallax_over_error.astype(float)
    motion = np.hypot(df.pmra.astype(float) / df.pmra_error.astype(float), df.pmdec.astype(float) / df.pmdec_error.astype(float))
    gaia_g = df.phot_g_mean_mag.astype(float)
    compact = df.mumax_minus_mag.astype(float)
    out["ra"] = df.coord_ra.astype(float)
    out["dec"] = df.coord_dec.astype(float)
    out["gaia_G"] = gaia_g
    out["parallax_snr"] = snr
    out["no_parallax"] = (snr.abs().fillna(0.0) < parallax_snr_max).astype(bool)
    out["motion_snr"] = motion
    out["no_motion"] = (motion.fillna(0.0) < motion_snr_max).astype(bool)
    out["moves"] = ((snr.fillna(0.0) > motion_snr_min) | (motion.fillna(0.0) > motion_snr_min)).astype(bool)
    out["gaia_quasar"] = df.in_qso_candidates.fillna(False).astype(bool)
    out["rubin_g_r"] = df.g_psfMag.astype(float) - df.r_psfMag.astype(float)
    y_h = pd.Series(_ab_mag(df.flux_y_templfit) - _ab_mag(df.flux_h_templfit), index=df.index)
    out["euclid_Y_H"] = y_h.where(gaia_g > bright_g)
    out["euclid_compactness"] = compact
    out["euclid_extended"] = (compact > extended_min).fillna(False).astype(bool)
    return out


def infrared_excess(df, sequence, on_sequence_mag=ON_SEQUENCE_MAG):
    """Runs on a table or on one partition: adds ``ir_excess``, the Euclid Y - H of each source minus that of
    stars with the same Rubin g - r (the star ``sequence`` from ``star_sequence``), and ``on_star_sequence``
    for sources within ``on_sequence_mag`` of the sequence."""
    out = df.copy()
    out["ir_excess"] = (df.euclid_Y_H.to_numpy(dtype=float)
                        - np.interp(df.rubin_g_r.to_numpy(dtype=float), sequence[0], sequence[1]))
    out["on_star_sequence"] = (out.ir_excess.abs() < on_sequence_mag).fillna(False).astype(bool)
    return out


def label(df):
    """Runs on one partition of selected objects: adds ``kind``, ``AGN`` for objects that pass ``AGN_CUT`` and
    ``star`` for the others."""
    out = df.copy()
    columns = [c for c in df.columns if c != "lightcurve"]
    out["kind"] = np.where(pd.DataFrame(df[columns]).eval(AGN_CUT).to_numpy(), "AGN", "star") if len(df) else ""
    return out


def selection_steps(sources, steps=AGN_STEPS):
    """Print how many ``sources`` remain after each cut of ``steps``, applied one after the other."""
    left = sources
    print(f"{'in Rubin, Gaia and Euclid':<52}{len(left):>8,}")
    for what, cut in steps:
        left = left.query(cut)
        print(f"{what:<52}{len(left):>8,}   {cut}")
    return left


def clean_lightcurves(df):
    """Runs on one partition: replaces the raw Rubin forced photometry of each object by its clean lightcurve
    (nested column ``lightcurve``: band, mjd, mag, mag_err; see ``epochs``) and adds ``rubin_epochs``, the
    number of clean epochs. Every object is kept."""
    import nested_pandas as npd

    index_name = df.index.name or "_healpix_29"
    base = pd.DataFrame(df[FEATURES]).reset_index()
    if len(df):
        ep = epochs(df)
    else:
        ep = pd.DataFrame({"star": np.array([], dtype=int), "band": np.array([], dtype=str), "mjd": np.array([], dtype=float),
                           "mag": np.array([], dtype=float), "mag_err": np.array([], dtype=float)})
    base["rubin_epochs"] = ep.groupby("star").size().reindex(base.index, fill_value=0).to_numpy().astype("int64")
    flat = ep.set_index("star")[["band", "mjd", "mag", "mag_err"]]
    flat.index.name = None
    out = npd.NestedFrame(base).join_nested(flat, "lightcurve")
    return out.set_index(index_name)


def as_catalog(table, partition_rows=SAMPLE_PARTITION_ROWS):
    """A table of objects (with nested lightcurves) as an LSDB catalog of about ``partition_rows`` objects per
    file, ready to be written with ``to_hats``."""
    return lsdb.from_dataframe(table.reset_index(drop=True), ra_column="ra", dec_column="dec",
                               partition_rows=partition_rows, highest_order=10, margin_threshold=None)


def lightcurve_table(sample):
    """The nested lightcurves of a computed catalog as one long table, with ``star`` = objectId."""
    table = sample.reset_index(drop=True)
    flat = table["lightcurve"].nest.to_flat()
    out = pd.DataFrame({"star": table.objectId.to_numpy()[flat.index.to_numpy()]})
    for name in ("band", "mjd", "mag", "mag_err"):
        out[name] = flat[name].to_numpy()
    out["band"] = out.band.astype(str)
    return out


def pair_sums(df, bands=SF_BANDS, lag_bins=LAG_BINS):
    """Runs on one partition: for each object and band, the pairs of epochs binned by the time between them.
    Returns one row per object, band and time bin with the number of pairs and the sums of (m_1 - m_2)^2,
    of the noise err_1^2 + err_2^2 and of the time difference. ``structure_function`` turns these into the
    structure function of a group."""
    edges = np.asarray(lag_bins, dtype=float)
    nbin = len(edges) - 1
    rows = []
    table = df.reset_index(drop=True)
    if len(table):
        flat = table["lightcurve"].nest.to_flat()
        flat = flat[flat["band"].astype(str).isin(bands)]
        ids, kinds = table.objectId.to_numpy(), table.kind.to_numpy()
        for (row, band), g in flat.groupby([flat.index, flat["band"].astype(str)]):
            t, m, e = (g[c].to_numpy(dtype=float) for c in ("mjd", "mag", "mag_err"))
            i, j = np.triu_indices(len(t), 1)
            lag = np.abs(t[j] - t[i])
            k = np.digitize(lag, edges) - 1
            ok = (k >= 0) & (k < nbin)
            count = np.bincount(k[ok], minlength=nbin)
            d2 = np.bincount(k[ok], weights=((m[j] - m[i]) ** 2)[ok], minlength=nbin)
            noise = np.bincount(k[ok], weights=(e[i] ** 2 + e[j] ** 2)[ok], minlength=nbin)
            lags = np.bincount(k[ok], weights=lag[ok], minlength=nbin)
            for b in np.flatnonzero(count):
                rows.append((ids[row], kinds[row], band, int(b), int(count[b]), d2[b], noise[b], lags[b]))
    out = pd.DataFrame(rows, columns=["objectId", "kind", "band", "bin", "pairs", "sum_d2", "sum_noise", "sum_lag"])
    return out.astype({"objectId": "int64", "kind": str, "band": str, "bin": "int64", "pairs": "int64",
                       "sum_d2": float, "sum_noise": float, "sum_lag": float})


def nested_lightcurves(ep, kind="AGN", redshift=None):
    """A long table of epochs (``star``, ``band``, ``mjd``, ``mag``, ``mag_err``) as one row per object with a
    nested ``lightcurve`` column, the layout ``pair_sums`` and ``rest_frame`` read. Every object gets the same
    ``kind``. ``redshift`` (a Series indexed by objectId) adds a ``redshift`` column."""
    import nested_pandas as npd

    ids = np.sort(ep.star.unique())
    base = pd.DataFrame({"objectId": ids, "kind": kind})
    if redshift is not None:
        base["redshift"] = redshift.reindex(ids).to_numpy(dtype=float)
    base = npd.NestedFrame(base)
    flat = ep[["band", "mjd", "mag", "mag_err"]].copy()
    flat.index = np.searchsorted(ids, ep.star.to_numpy())
    return base.join_nested(flat.sort_index(), "lightcurve")


def equal_pairs(sums, points=SF_POINTS, kind="AGN"):
    """Merge the fine time bins of ``pair_sums`` so that every point of the structure function has about
    the same number of pairs. Done per band: the fine bins are walked in order of time and cut where the
    running number of pairs of the ``kind`` objects passes 1/``points``, 2/``points``, ... of the total.
    A fine bin is never split, so the numbers are equal only roughly. Other kinds get the same bins."""
    out = sums.copy()
    for band, rows in sums[sums.kind == kind].groupby("band"):
        per_bin = rows.groupby("bin").pairs.sum().sort_index()
        target, filled, current, labels = per_bin.sum() / points, 0, 0, []
        for n in per_bin.to_numpy():  # close a point when it has its share and enough pairs remain for another
            labels.append(current)
            filled += n
            if filled >= target and current < points - 1:
                current, filled = current + 1, 0
        merged = pd.Series(labels, index=per_bin.index)
        if filled < target / 2 and current > 0:  # a small remainder joins the point before it
            merged[merged == current] = current - 1
        here = out.band == band
        out.loc[here, "bin"] = out.loc[here, "bin"].map(merged).fillna(-1).astype("int64")
    out = out[out.bin >= 0]
    return out.groupby(["objectId", "kind", "band", "bin"], as_index=False)[["pairs", "sum_d2", "sum_noise", "sum_lag"]].sum()


def structure_function_of(sums, kind, band, remove_noise=False, min_pairs=3, min_sources=5, n_boot=200, seed=0,
                          ids=None):
    """Structure function of the objects of one ``kind`` in one ``band`` from the table of ``pair_sums``.

    For each object and time bin: the mean of (m_1 - m_2)^2 over its pairs; with ``remove_noise`` the noise
    the reported errors predict for the same pairs is subtracted. The group value ``sf2`` is the median over
    objects (each counts once; an object needs ``min_pairs`` pairs in the bin, a bin needs ``min_sources``
    objects). ``sf_mag`` is its square root, negative when the errors predict more scatter than is seen.
    ``err`` is the spread over ``n_boot`` resamplings of the objects. ``ids`` restricts to those objects."""
    s = sums[(sums.kind == kind) & (sums.band == band) & (sums.pairs >= min_pairs)]
    if ids is not None:
        s = s[s.objectId.isin(ids)]
    value = (s.sum_d2 - (s.sum_noise if remove_noise else 0.0)) / s.pairs
    per_source = value.groupby([s.objectId, s.bin]).first().unstack("bin")
    root = lambda v: np.sign(v) * np.sqrt(np.abs(v))  # noqa: E731
    rng = np.random.default_rng(seed)
    values = per_source.to_numpy()
    enough = np.isfinite(values).sum(axis=0) >= min_sources

    def group(rows):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.where(enough, np.nanmedian(values[rows], axis=0), np.nan)

    everyone = np.arange(len(values))
    boot = [root(group(rng.choice(everyone, len(everyone)))) for _ in range(n_boot)] if len(values) else [np.full(values.shape[1], np.nan)]
    totals = s.groupby("bin")[["sum_lag", "pairs"]].sum().reindex(per_source.columns)
    sf2 = group(everyone)
    return pd.DataFrame({"lag_days": (totals.sum_lag / totals.pairs).to_numpy(), "sf2": sf2, "sf_mag": root(sf2),
                         "err": np.nanstd(boot, axis=0), "sources": np.isfinite(values).sum(axis=0),
                         "pairs": totals.pairs.to_numpy()})


def structure_functions_of(sums, labels, remove_noise=False, bands=SF_BANDS, ids=None):
    """``structure_function_of`` for every band and every kind in ``labels`` (kind -> label for the plot):
    a dict band -> {label: table}, ready for ``plot_structure_function``."""
    return {band: {label: structure_function_of(sums, kind, band, remove_noise=remove_noise, ids=ids)
                   for kind, label in labels.items()} for band in bands}


# ---------------------------------------------------------------------------------------------------------
# Redshifts from Milliquas, and the rest frame
# ---------------------------------------------------------------------------------------------------------
MILLIQUAS_URL = "https://quasars.org/milliquas.fits.zip"
MILLIQUAS_DIR = os.path.join(RESULTS_DIR, "milliquas")
# Objects per file of the Milliquas catalog built in memory.
MILLIQUAS_PARTITION_ROWS = 500
MJD_ZERO = 60000.0


def milliquas(region=None, url=MILLIQUAS_URL):
    """The Million Quasars catalog (Milliquas v8, Flesch 2023) as a plain pandas table: position, name,
    type and redshift. It is one FITS file from quasars.org, not a HATS catalog; it is downloaded the first
    time and kept. ``region`` (a cone) keeps the rows in a box around it."""
    from astropy.table import Table

    path = os.path.join(MILLIQUAS_DIR, "milliquas.fits")
    if not os.path.exists(path):
        import io
        import urllib.request
        import zipfile

        os.makedirs(MILLIQUAS_DIR, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).extractall(MILLIQUAS_DIR)
    t = Table.read(path)[["RA", "DEC", "NAME", "TYPE", "Z"]].to_pandas()
    if region is not None:
        r = region.radius_arcsec / 3600 + 0.1
        t = t[(np.abs(t.DEC - region.dec) < r) & (np.abs(t.RA - region.ra) < r / np.cos(np.radians(region.dec)))]
    text = lambda c: c.map(lambda v: v.decode() if isinstance(v, bytes) else str(v)).str.strip()  # noqa: E731
    return pd.DataFrame({"mq_ra": t.RA.astype(float), "mq_dec": t.DEC.astype(float), "mq_name": text(t.NAME),
                         "mq_type": text(t.TYPE), "redshift": t.Z.astype(float)}).reset_index(drop=True)


def rest_frame(df, mjd_zero=MJD_ZERO):
    """Runs on one partition: replaces the time of every epoch by the time in the rest frame of its source,
    (MJD - ``mjd_zero``) / (1 + redshift) [days]. A source at redshift 1 is seen in slow motion by a factor of
    two; this undoes it. The partition needs the columns ``redshift`` and the nested ``lightcurve``."""
    out = df.copy()
    if len(out):
        lengths = np.asarray(out["lightcurve"].array.list_lengths)
        z = np.repeat(out.redshift.to_numpy(dtype=float), lengths)
        mjd = out["lightcurve"].nest.to_flat().mjd.to_numpy(dtype=float)
        out["lightcurve.mjd"] = (mjd - mjd_zero) / (1.0 + z)
    return out


def plot_catalog(catalog, table, cone=None, others=None, title="", tile="file of the matched catalog",
                 value="redshift", value_label="redshift"):
    """What a catalog looks like: its files as tiles and its objects (``table``, the computed catalog) as
    points coloured by ``value``. ``others`` (name -> catalog) draws the files of other catalogs."""
    from IPython.display import display

    picture = _TileMap(catalog.get_healpix_pixels(), cone, others=others, tile=tile, value_label=value_label)
    picture.ax.scatter(table.ra, table.dec, c=table[value], cmap="plasma", vmin=picture.value_range[0],
                       vmax=picture.value_range[1], s=45, edgecolor="k", lw=0.5, zorder=5)
    handles, names = picture.ax.get_legend_handles_labels()
    from matplotlib.patches import Patch

    picture.ax.legend(handles=[Patch(facecolor="0.82", edgecolor="tab:blue", label=tile), *handles], loc="upper left",
                      ncol=2, frameon=True, framealpha=0.95, edgecolor="0.3", fontsize=12)
    picture.ax.set_title(title, pad=12, fontsize=16)
    display(picture.fig)


# ---------------------------------------------------------------------------------------------------------
# A diagram of how HATS and LSDB work
# ---------------------------------------------------------------------------------------------------------
def plot_how_it_works():
    """A diagram: what is on disk in a HATS catalog (top) and what LSDB does with it (bottom)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(20, 9.5))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=color, edgecolor="0.25", lw=2))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=16, fontweight="bold")
        ax.text(x + 0.2, y + h - 0.85, body, ha="left", va="top", fontsize=13, family="monospace", linespacing=1.35)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=28, lw=2.5, color="0.25"))

    ax.text(0.1, 9.3, "HATS: what is on disk", fontsize=19, fontweight="bold", va="top")
    box(0.2, 5.3, 6.2, 3.3, "one catalog = one directory",
        "gaia_dr3/\n  hats.properties     name, sizes\n  partition_info.csv  list of tiles\n"
        "  dataset/\n    Norder=2/Dir=0/Npix=71.parquet\n    Norder=3/Dir=0/Npix=290.parquet", "#dbe9f6")
    box(6.9, 5.3, 6.2, 3.3, "one file = one HEALPix tile",
        "crowded sky  -> small tiles\nempty sky    -> large tiles\nso files have similar sizes.\n"
        "Rows are sorted by position.\nNested columns hold lightcurves.", "#dbe9f6")
    box(13.6, 5.3, 6.2, 3.3, "every catalog, the same grid",
        "A tile of one catalog lines up\nwith tiles of every other one.\nA margin file per tile holds the\n"
        "neighbours just outside it, so\nmatches at the edges are found.", "#dbe9f6")

    ax.text(0.1, 4.6, "LSDB: what the code does", fontsize=19, fontweight="bold", va="top")
    steps = [("1. open", "lsdb.open_catalog(\n  path_or_url,\n  columns=[...],\n  search_filter=cone)\nreads the list of\ntiles, no rows", "#e3f2df"),
             ("2. plan (lazy)", "a.crossmatch(b)\n.query('...')\n.map_partitions(f)\nnothing runs yet:\none task per group\nof aligned tiles", "#e3f2df"),
             ("3. run", ".compute()\nDask workers each\ntake one task: read\nthe Parquet files of\nits tiles, match,\nfilter, apply f", "#fdebd0"),
             ("4. result", "a pandas table\n(with nested\ncolumns), or\n.to_hats(path):\na new catalog on\nthe same grid", "#fdebd0")]
    w, gap = 4.45, 0.55
    for k, (title, body, color) in enumerate(steps):
        x = 0.2 + k * (w + gap)
        box(x, 0.3, w, 3.6, title, body, color)
        if k:
            arrow(x - gap + 0.08, x - 0.08, 2.1)
    return fig
