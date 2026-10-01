import json
import os
import glob
import gsw 

import cartopy.feature as cfeature
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata, interp1d
from scipy.stats import linregress
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    box,
)

import matplotlib.pyplot as plt


def load_variable_metadata(metadata_path):
    """
    Load variable metadata from a JSON file.
    """
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_cruise_datasets(cruise, transect_name, variable_metadata):
    """
    Load datasets for a given cruise and transect name based on variable metadata.
    """
    base = os.environ["OMIPATH"]
    wkdir = f"{base}/Coastal Measurements/{cruise}/EcoCTD/data/{transect_name}/Level2/"

    dataset_types = sorted({meta["dataset"] for meta in variable_metadata.values()})
    datasets_by_type = {}

    for data_type in dataset_types:
        files = sorted(glob.glob(os.path.join(wkdir, f"*{data_type}.nc")))
        datasets_by_type[data_type] = [xr.open_dataset(f) for f in files]

    return datasets_by_type


def build_profile_matrix(
    datasets,
    var,
    depth_grid,
    meta,
    do_interpolate=True,
    extrapolate=True,
    ignore_questionable=True,
):
    profiles = np.full((len(depth_grid), len(datasets)), np.nan)

    for i, ds in enumerate(datasets):
        if "seaPress" not in ds:
            continue

        P = ds["seaPress"].values

        if var not in ds and var != "rho":
            continue

        if var == "rho":
            if "SA" not in ds or "CT" not in ds:
                continue
            SA = ds["SA"].values
            CT = ds["CT"].values
            V = gsw.pot_rho_t_exact(SA, CT, P, 0)
        else:
            V = ds[var].values
            if var == "O2":
                V = V * 1000.0

        qc_mask = np.ones_like(V, dtype=bool)
        qc_var = f"{var}_qc"
        if qc_var in ds:
            QC = ds[qc_var].values
            if ignore_questionable:
                qc_mask = QC == 1
            else:
                qc_mask = (QC == 1) | (QC == 0)

        manual_mask = np.ones_like(V, dtype=bool)
        
        vmin, vmax = meta[var]["vlim"]
        manual_mask = (V >= vmin) & (V <= vmax)

        valid_mask = (~np.isnan(P)) & (~np.isnan(V)) & qc_mask & manual_mask
        P = P[valid_mask]
        V = V[valid_mask]

        if P.size < 2:
            continue

        if do_interpolate:
            f = interp1d(P, V, kind="linear", bounds_error=False)
            profiles[:, i] = f(depth_grid)
        else:
            for j, depth in enumerate(P):
                k = np.argmin(np.abs(depth_grid - depth))
                profiles[k, i] = V[j]

    if extrapolate:
        n_depth, n_profiles = profiles.shape
        for k in range(n_depth):
            row = profiles[k, :]
            valid_idx = np.where(np.isfinite(row))[0]
            if valid_idx.size == 0:
                continue
            f_nearest = interp1d(
                valid_idx,
                row[valid_idx],
                kind="nearest",
                bounds_error=False,
                fill_value="extrapolate",
            )
            profiles[k, :] = f_nearest(np.arange(n_profiles))

    return profiles



def plot_sections_grid(
    cruises,
    transect_name,
    variables,
    metadata_path="variable_metadata.json",
    shared_xlim=(15, 20),
    depth_limit=100,
    do_interpolate=True,
    extrapolate=True,
    ignore_questionable=True,
    include_topo=True,
    use_density_contours=True,
    coast_lon=None,
    coast_lat=None, 
):
    variable_metadata = load_variable_metadata(metadata_path)

    nrows = len(cruises)
    ncols = len(variables)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.0 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    depth_grid = np.linspace(5, depth_limit, depth_limit)

    for r, cruise in enumerate(cruises):
        datasets_by_type = load_cruise_datasets(cruise, transect_name, variable_metadata)

        # Use CTD profiles to define distance and (optionally) topography
        ctd_datasets = datasets_by_type.get("ctd", [])
        if len(ctd_datasets) == 0:
            continue

        # Assumes you already split these out:
        # - calc_cross_shore_distance(...) -> geometry only
        # - extract_transect_bathymetry(...) -> topo only
        geom = calc_cross_shore_distance(ctd_datasets, coast_lon, coast_lat)
        distance = geom["distance"]
        profile_lats = geom["profile_lats"]
        profile_lons = geom["profile_lons"]

        transect_depths = None
        if include_topo:
            transect_depths = extract_transect_bathymetry(
                profile_lons=profile_lons,
                profile_lats=profile_lats,
                region=[-3, 2, 2, 7],   # change if needed
                depth_jump_threshold=150,
            )

        # density contours from CTD, once per cruise
        if use_density_contours:
            rho_profiles = build_profile_matrix(
                ctd_datasets,
                "rho",
                depth_grid,
                variable_metadata,
                do_interpolate=do_interpolate,
                extrapolate=extrapolate,
                ignore_questionable=ignore_questionable,
            )

        for c, var in enumerate(variables):
            ax = axs[r, c]
            meta = variable_metadata[var]
            data_type = meta["dataset"]

            datasets = datasets_by_type.get(data_type, [])
            if len(datasets) == 0:
                ax.set_axis_off()
                continue

            profiles = build_profile_matrix(
                datasets,
                var,
                depth_grid,
                variable_metadata,
                do_interpolate=do_interpolate,
                extrapolate=extrapolate,
                ignore_questionable=ignore_questionable,
            )

            # color limits from metadata JSON
            vmin, vmax = meta["vlim"]

            # special case: chlorophyll plotted on log10 scale
            if var == "chl_cal":
                profiles = np.where(profiles > 0, np.log10(profiles), np.nan)
                vmin = np.log10(max(vmin, 1e-3))
                vmax = np.log10(max(vmax, 1e-3))

            X, Y = np.meshgrid(distance, depth_grid)

            mesh = ax.pcolormesh(
                X,
                Y,
                profiles,
                shading="auto",
                cmap=meta["cmap"],
                vmin=vmin,
                vmax=vmax,
                zorder=1,
            )

            # optional density contours
            if use_density_contours:
                contour_data = np.ma.masked_invalid(rho_profiles)
                cs = ax.contour(
                    X,
                    Y,
                    contour_data,
                    colors="black",
                    linewidths=0.5,
                    levels=np.arange(1021, 1027.25, 0.25),
                    zorder=2,
                )
                ax.contour(
                    X,
                    Y,
                    contour_data,
                    levels=[1026],
                    colors="black",
                    linewidths=1.0,
                    zorder=3,
                )

            # optional topography
            if include_topo and transect_depths is not None:
                topo_arr = np.array(transect_depths, dtype=float)
                if topo_arr[0] < 0:
                    topo_arr = -topo_arr

                ax.plot(distance, topo_arr, color="k", linewidth=0.75, zorder=4)
                mask = np.isfinite(topo_arr)
                ax.fill_between(
                    distance[mask],
                    topo_arr[mask],
                    y2=depth_limit,
                    color="lightgray",
                    alpha=1,
                    zorder=3,
                )

            # panel formatting
            ax.set_xlim(*shared_xlim)
            ax.set_ylim(depth_limit, 0)

            if r == 0:
                ax.set_title(f"{var}  ({meta['dataset']})")

            if c == 0:
                ax.set_ylabel(f"{cruise}\nPressure (dbar)")
            else:
                ax.set_ylabel("")

            cbar = plt.colorbar(mesh, ax=ax, pad=0.01)

            if var == "chl_cal":
                cbar.set_label(meta["label"])
                ticks = [0.1, 1.0, 10.0]
                cbar.set_ticks(np.log10(ticks))
                cbar.set_ticklabels([str(t) for t in ticks])

            elif var == "bb470_cal":
                cbar.set_label(meta["label"])
                ticks = np.array([0.0, 0.5e-3, 1.0e-3, 1.5e-3])
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{t*1e3:.1f}" for t in ticks])

            else:
                cbar.set_label(meta["label"])

            if r == nrows - 1:
                ax.set_xlabel("Distance from Shore (km)")
            

    return fig, axs


def coastline_points(region, scale="10m"):
    """
    Extracts the coastline coordinates from the Natural Earth database using cartopy
    
    Returns the lat and lon coordinates
    """

    lon_min, lon_max, lat_min, lat_max = region
    clip_box = box(lon_min, lat_min, lon_max, lat_max)

    coast = cfeature.NaturalEarthFeature("physical", "coastline", scale)

    lon_all = []
    lat_all = []

    for geom in coast.geometries():
        clipped = geom.intersection(clip_box)

        if clipped.is_empty:
            continue

        # Handle the different geometry types that can come back
        geoms = []
        if clipped.geom_type == "LineString":
            geoms = [clipped]
        elif clipped.geom_type == "MultiLineString":
            geoms = list(clipped.geoms)
        elif clipped.geom_type == "GeometryCollection":
            geoms = [g for g in clipped.geoms if g.geom_type == "LineString"]

        for g in geoms:
            coords = np.asarray(g.coords)
            lon_all.append(coords[:, 0])
            lat_all.append(coords[:, 1])

    if not lon_all:
        return np.array([]), np.array([])

    return np.concatenate(lon_all), np.concatenate(lat_all)


def calc_cross_shore_distance(datasets, coast_lon, coast_lat):
    """
    Compute along-transect cross-shore distance for each profile station.

    Always returns stations ordered inshore -> offshore.
    """
    # ----------------------------
    # 1. Extract profile coordinates
    # ----------------------------
    coords = [(ds.lat[0].values.item(), ds.lon[0].values.item()) for ds in datasets]
    profile_lats, profile_lons = zip(*coords)
    profile_lats = np.asarray(profile_lats, dtype=float)
    profile_lons = np.asarray(profile_lons, dtype=float)

    if len(profile_lats) < 2:
        raise ValueError("Need at least two profiles to define a transect.")

    # ----------------------------
    # 2. Best-fit transect line in lon-lat space
    # ----------------------------
    slope, intercept, _, _, _ = linregress(profile_lons, profile_lats)

    transect_lons = np.linspace(
        np.min(profile_lons) - 0.1,
        np.max(profile_lons) + 0.1,
        400,
    )
    transect_lats = slope * transect_lons + intercept

    # ----------------------------
    # 3. Find all coastline intersections
    # ----------------------------
    intersections = []

    for i in range(len(transect_lons) - 1):
        p1 = (transect_lons[i], transect_lats[i])
        p2 = (transect_lons[i + 1], transect_lats[i + 1])

        for j in range(len(coast_lon) - 1):
            q1 = (coast_lon[j], coast_lat[j])
            q2 = (coast_lon[j + 1], coast_lat[j + 1])

            if do_lines_intersect(p1, p2, q1, q2):
                try:
                    inter_lon, inter_lat = line_intersection(p1, p2, q1, q2)
                    if np.isfinite(inter_lon) and np.isfinite(inter_lat):
                        intersections.append((inter_lon, inter_lat))
                except Exception:
                    pass

    if len(intersections) == 0:
        raise ValueError("No intersection found between transect line and coastline.")

    intersections = np.asarray(intersections)

    # Choose the coastline intersection closest to the profile centroid
    centroid_lon = np.mean(profile_lons)
    centroid_lat = np.mean(profile_lats)
    d2 = (intersections[:, 0] - centroid_lon) ** 2 + (intersections[:, 1] - centroid_lat) ** 2
    best_idx = np.argmin(d2)
    intersect_lon, intersect_lat = intersections[best_idx]

    # ----------------------------
    # 4. Build unit transect direction vector in km
    # ----------------------------
    ref1 = (transect_lats[0], transect_lons[0])
    ref2 = (transect_lats[-1], transect_lons[-1])

    dx = haversine_np((ref1[0], ref1[1]), (ref1[0], ref2[1]))
    if ref2[1] < ref1[1]:
        dx *= -1

    dy = haversine_np((ref1[0], ref1[1]), (ref2[0], ref1[1]))
    if ref2[0] < ref1[0]:
        dy *= -1

    transect_vec = np.array([dx, dy], dtype=float)
    norm = np.linalg.norm(transect_vec)
    if norm == 0:
        raise ValueError("Transect direction vector has zero length.")
    transect_vec /= norm

    # ----------------------------
    # 5. Project each profile onto the transect
    # ----------------------------
    original_distance = []
    for lat, lon in zip(profile_lats, profile_lons):
        dlon = haversine_np((intersect_lat, intersect_lon), (intersect_lat, lon))
        if lon < intersect_lon:
            dlon *= -1

        dlat = haversine_np((intersect_lat, intersect_lon), (lat, intersect_lon))
        if lat < intersect_lat:
            dlat *= -1

        vec = np.array([dlon, dlat], dtype=float)
        original_distance.append(np.dot(vec, transect_vec))

    original_distance = np.asarray(original_distance)

    # ----------------------------
    # 6. Force inshore -> offshore ordering
    # ----------------------------
    distance = np.abs(original_distance)
    sort_idx = np.argsort(distance)

    profile_lats = profile_lats[sort_idx]
    profile_lons = profile_lons[sort_idx]
    distance = distance[sort_idx]

    coords = [coords[i] for i in sort_idx]
    datasets = [datasets[i] for i in sort_idx]

    return {
        "profile_lats": profile_lats,
        "profile_lons": profile_lons,
        "distance": distance,
        "transect_lons": transect_lons,
        "transect_lats": transect_lats,
        "intersect_lon": intersect_lon,
        "intersect_lat": intersect_lat,
        "coords": coords,
        "datasets": datasets,
    }

def line_intersection(p1, p2, q1, q2):
    # Extracting coordinates
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    
    # Denominator for the intersection
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Check if the lines are parallel (denom == 0)
    if denom == 0:
        return None  # No intersection (parallel lines)
    
    # Calculate the intersection (t, u) parameters
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    
    # Calculate the intersection point
    intersect_x = x1 + t * (x2 - x1)
    intersect_y = y1 + t * (y2 - y1)
    
    return intersect_x, intersect_y
    

def do_lines_intersect(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def haversine_np(coord1, coord2):
    """
    Calculate the great-circle distance between two points 
    on the earth (specified in decimal degrees), given as (lat, lon) tuples.

    Args:
        coord1: tuple (lat1, lon1)
        coord2: tuple (lat2, lon2)

    Returns:
        Distance in kilometers.
    """
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    km = 6367 * c  # Earth's radius in km
    return km


def interpolate_nans(data, x):
    """Interpolates over NaN values in 1D data."""
    mask = ~np.isnan(data)  # Find valid (non-NaN) values
    if np.sum(mask) < 2:  # Need at least two points for interpolation
        return data  # Return original array if not enough valid data
    interp_func = interp1d(x[mask], data[mask], kind='linear', bounds_error=False, fill_value="extrapolate")
    return interp_func(x)


def extract_transect_bathymetry(profile_lons, profile_lats, region, bathy_ds=None, bathy_file=None,
                                depth_jump_threshold=150):
    """
    Sample bathymetry at the profile locations.

    Returns a 1D array of depths aligned with profile_lons/profile_lats.
    """
    close_ds = False

    if bathy_ds is None:
        if bathy_file is not None:
            bathy_ds = xr.open_dataset(bathy_file)
            close_ds = True
        else:
            bathy_ds = xr.open_dataset("GIS_data/gebco_2023_n12.0_s-12.0_w-21.0_e16.0.nc")
            close_ds = True

    try:
        lon_min, lon_max = min(region[0], region[1]), max(region[0], region[1])
        lat_min, lat_max = min(region[2], region[3]), max(region[2], region[3])

        if bathy_ds.lat.values[0] < bathy_ds.lat.values[-1]:
            lat_slice = slice(lat_min, lat_max)
        else:
            lat_slice = slice(lat_max, lat_min)

        bathy_ds = bathy_ds.sel(lon=slice(lon_min, lon_max), lat=lat_slice)

        b_lats = bathy_ds["lat"].values
        b_lons = bathy_ds["lon"].values
        b_depth_grid = bathy_ds["elevation"].values

        lon2d, lat2d = np.meshgrid(b_lons, b_lats)
        b_lons_flat = lon2d.ravel()
        b_lats_flat = lat2d.ravel()
        b_depths_flat = b_depth_grid.ravel()

        valid_mask = np.isfinite(b_depths_flat)
        b_lons_flat = b_lons_flat[valid_mask]
        b_lats_flat = b_lats_flat[valid_mask]
        b_depths_flat = b_depths_flat[valid_mask]

        transect_depths = griddata(
            (b_lons_flat, b_lats_flat),
            b_depths_flat,
            (profile_lons, profile_lats),
            method="nearest",
        )

        # Optional jump filtering
        if transect_depths is not None and len(transect_depths) > 1:
            valid_depth_mask = np.isfinite(transect_depths)

            if np.sum(valid_depth_mask) > 1:
                valid_depths = transect_depths[valid_depth_mask]
                depth_changes = np.abs(np.diff(valid_depths))
                jump_index = np.where(depth_changes > depth_jump_threshold)[0]

                if jump_index.size > 0:
                    first_jump_valid = jump_index[0] + 1
                    valid_positions = np.where(valid_depth_mask)[0]
                    first_jump_full = valid_positions[first_jump_valid]
                    transect_depths[first_jump_full:] = np.nan

        return transect_depths

    finally:
        if close_ds:
            bathy_ds.close()