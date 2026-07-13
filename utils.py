"""
Contains:
    > haversine_np(coord1, coord2) 
    > interpolate_nans(data, x)
    > coastline_extraction(lonmin, lonmax, latmin, latmax, coast_file)
    > extract_polygon_coords(geom, min_area_threshold)
    > line_intersection(p1, p2, q1, q2)
    > do_lines_intersect(p1, p2, q1, q2)
    > fill_nan_2d(grid)
"""
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from scipy.interpolate import interp1d, griddata
from scipy.stats import linregress

def calc_cross_shore_distance_original(datasets, datasets_ctd, coast_lon, coast_lat, region, gebco_bathy=True):

    # extract profile coordinates
    coords = [(ds.lat[0].values.item(), ds.lon[0].values.item()) for ds in datasets]

    profile_lats, profile_lons = zip(*coords)
    profile_lats = np.array(profile_lats)
    profile_lons = np.array(profile_lons)

    # define a line of best fit through the profile coordinates
    slope, intercept, _, _, _ = linregress(profile_lons, profile_lats)

    def transect_line(lon):
        return slope * lon + intercept

    coast_lons = np.array(coast_lon)  
    coast_lats = np.array(coast_lat)  

    transect_lons = np.linspace(min(profile_lons) - 0.1, max(profile_lons) + 0.1, 100)  # Extended range
    transect_lats = slope * transect_lons + intercept


    # finds the intersection of two line segments p1p2 and q1q2
    for i in range(len(transect_lons) - 1):
        # Get the transect segment
        p1 = (transect_lons[i], transect_lats[i])
        p2 = (transect_lons[i+1], transect_lats[i+1])

        # Loop over each coastline segment
        for j in range(len(coast_lon) - 1):
            # Get the coastline segment
            q1 = (coast_lon[j], coast_lat[j])
            q2 = (coast_lon[j+1], coast_lat[j+1])

            # Check if the transect segment intersects with the coastline segment
            if do_lines_intersect(p1, p2, q1, q2):
                print(f"Intersection found between transect segment {i} and coastline segment {j}")

                transect_p1 = (transect_lons[i], transect_lats[i])
                transect_p2 = (transect_lons[i+1], transect_lats[i+1])
                coast_q1 = (coast_lon[j], coast_lat[j])
                coast_q2 = (coast_lon[j+1], coast_lat[j+1])
                
                intersect_lon, intersect_lat = line_intersection(transect_p1, transect_p2, coast_q1, coast_q2)
                print(f"Intersection at: {(intersect_lon, intersect_lat)}")


    # --- Step 1. Build transect direction vector in km ---
    ref1 = (transect_lats[0], transect_lons[0])
    ref2 = (transect_lats[-1], transect_lons[-1])

    dx = haversine_np((ref1[0], ref1[1]), (ref1[0], ref2[1]))
    if ref2[1] < ref1[1]:
        dx *= -1

    dy = haversine_np((ref1[0], ref1[1]), (ref2[0], ref1[1]))
    if ref2[0] < ref1[0]:
        dy *= -1

    transect_vec = np.array([dx, dy])
    transect_vec = transect_vec / np.linalg.norm(transect_vec)

    # --- Step 2. Compute along-transect distance in km for each profile ---
    original_distance = []
    for lat, lon in zip(profile_lats, profile_lons):
        dlon = haversine_np((intersect_lat, intersect_lon), (intersect_lat, lon))
        if lon < intersect_lon:
            dlon *= -1

        dlat = haversine_np((intersect_lat, intersect_lon), (lat, intersect_lon))
        if lat < intersect_lat:
            dlat *= -1

        vec = np.array([dlon, dlat])
        original_distance.append(np.dot(vec, transect_vec))

    original_distance = np.array(original_distance)

    # --- Step 3. Sort profiles ---
    sort_idx = np.argsort(original_distance)
    profile_lats   = profile_lats[sort_idx]
    profile_lons   = profile_lons[sort_idx]
    distance = original_distance[sort_idx]

    # --- Step 4. Safeguard: make sure first profile is the most inshore one ---
    # Check which end is closer to the coastline intersection (0 km)
    if abs(distance[-1]) < abs(distance[0]):
        profile_lats = profile_lats[::-1]
        profile_lons = profile_lons[::-1]
        distance = distance[::-1]


    if gebco_bathy:

        df = xr.open_dataset('GIS_data/gebco_2023_n12.0_s-12.0_w-21.0_e16.0.nc')
        
        df = df.sel(lon = slice(region[0], region[1]), lat=slice(region[2], region[3]))
        b_lats = df["lat"].values
        b_lons = df["lon"].values
        d_depths = df["elevation"].values

        b_lons, b_lats = np.meshgrid(b_lons, b_lats)
        
        # Flatten all three
        b_lons = b_lons.ravel()
        b_lats = b_lats.ravel()
        b_depths = d_depths.ravel()
        
    else:
        bathy_file = '/vortexfs1/share/mahadevanlab/datasets/OMI_pilot_campaign_jan2025/bathymetry_data.csv'
        df = pd.read_csv(bathy_file, skiprows=1, names=["time","lat", "lon", "depth","temp"])
        
        b_lats = df["lat"].values
        b_lons = df["lon"].values
        b_depths = df["depth"].values
        
        # remove nans
        valid_mask = ~np.isnan(b_depths)
        b_lons, b_lats, b_depths = b_lons[valid_mask], b_lats[valid_mask], b_depths[valid_mask]
        
    transect_depths = griddata(
        (b_lons, b_lats), b_depths, (profile_lons, profile_lats), method='nearest'
    )

    # filter out points off the shelf
    depth_changes = np.abs(np.diff(transect_depths))

    jump_index = np.where(depth_changes > 150)[0]

    if jump_index.size > 0: 
        first_jump = jump_index[0] + 1  # +1 to include the first invalid point
        transect_depths[first_jump:] = np.nan

    coords = [coords[i] for i in sort_idx]
    datasets = [datasets[i] for i in sort_idx]
    datasets_ctd = [datasets_ctd[i] for i in sort_idx] 

    return {
        "profile_lats": profile_lats,
        "profile_lons": profile_lons,
        "distance": distance,
        "transect_depths": transect_depths,
        "transect_lons": transect_lons,
        "transect_lats": transect_lats,
        "intersect_lon": intersect_lon,
        "intersect_lat": intersect_lat,
        "coords": coords,
        "datasets": datasets,
        "datasets_ctd": datasets_ctd,
    }


def calc_cross_shore_distance(
    datasets,
    datasets_ctd,
    coast_lon,
    coast_lat,
    region,
    gebco_bathy=True,
    depth_jump_threshold=150,
):
    """
    Compute cross-shore distance of profile stations along a mostly linear transect,
    referenced to the coastline intersection.

    Key fixes in this version:
    - Collects all coastline intersections and chooses the one nearest the profile centroid.
    - Applies one final station ordering consistently to profiles, coords, datasets, and bathy.
    - Handles GEBCO latitude slice direction safely.
    - Fails clearly if no coastline intersection is found.
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
                    # ignore degenerate segments / numerical edge cases
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
    # 5. Project each profile onto the transect, relative to coastline intersection
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
    # 6. Sort profiles by projected distance
    #    Then ensure first point is the one nearest the coast intersection
    # ----------------------------
    sort_idx = np.argsort(original_distance)
    sorted_distance = original_distance[sort_idx]

    # Make ordering nearshore -> offshore based on closeness to coastline intersection
    if np.abs(sorted_distance[-1]) < np.abs(sorted_distance[0]):
        sort_idx = sort_idx[::-1]

    profile_lats = profile_lats[sort_idx]
    profile_lons = profile_lons[sort_idx]
    distance = original_distance[sort_idx]

    coords = [coords[i] for i in sort_idx]
    datasets = [datasets[i] for i in sort_idx]
    datasets_ctd = [datasets_ctd[i] for i in sort_idx]

    # Optional: reset distance so coastline is 0 and first station is smallest offshore distance
    # This makes interpretation a bit cleaner.
    if distance[0] < 0 and distance[-1] < 0:
        distance = -distance

    # ----------------------------
    # 7. Load bathymetry
    # ----------------------------
    if gebco_bathy:
        bathy_ds = xr.open_dataset("GIS_data/gebco_2023_n12.0_s-12.0_w-21.0_e16.0.nc")

        lon_min, lon_max = min(region[0], region[1]), max(region[0], region[1])
        lat_min, lat_max = min(region[2], region[3]), max(region[2], region[3])

        # Handle ascending vs descending latitude in the source grid
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

    else:
        bathy_file = "/vortexfs1/share/mahadevanlab/datasets/OMI_pilot_campaign_jan2025/bathymetry_data.csv"
        bathy_df = pd.read_csv(
            bathy_file,
            skiprows=1,
            names=["time", "lat", "lon", "depth", "temp"],
        )

        b_lats_flat = bathy_df["lat"].values
        b_lons_flat = bathy_df["lon"].values
        b_depths_flat = bathy_df["depth"].values

        valid_mask = (
            np.isfinite(b_lats_flat)
            & np.isfinite(b_lons_flat)
            & np.isfinite(b_depths_flat)
        )
        b_lons_flat = b_lons_flat[valid_mask]
        b_lats_flat = b_lats_flat[valid_mask]
        b_depths_flat = b_depths_flat[valid_mask]

    # ----------------------------
    # 8. Sample bathymetry at sorted profile locations
    # ----------------------------
    transect_depths = griddata(
        (b_lons_flat, b_lats_flat),
        b_depths_flat,
        (profile_lons, profile_lats),
        method="nearest",
    )

    # ----------------------------
    # 9. Filter points after first unrealistic jump
    #    Assumes nearshore -> offshore ordering
    # ----------------------------
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

    return {
        "profile_lats": profile_lats,
        "profile_lons": profile_lons,
        "distance": distance,
        "transect_depths": transect_depths,
        "transect_lons": transect_lons,
        "transect_lats": transect_lats,
        "intersect_lon": intersect_lon,
        "intersect_lat": intersect_lat,
        "coords": coords,
        "datasets": datasets,
        "datasets_ctd": datasets_ctd,
    }

def calc_cross_shore_distance_argo(datasets, coast_lon, coast_lat, region, gebco_bathy=True):

    # extract profile coordinates
    coords = [(ds.lat[0].values.item(), ds.lon[0].values.item()) for ds in datasets]

    profile_lats, profile_lons = zip(*coords)
    profile_lats = np.array(profile_lats)
    profile_lons = np.array(profile_lons)

    # define a line of best fit through the profile coordinates
    slope, intercept, _, _, _ = linregress(profile_lons, profile_lats)

    def transect_line(lon):
        return slope * lon + intercept

    coast_lons = np.array(coast_lon)  
    coast_lats = np.array(coast_lat)  

    transect_lons = np.linspace(min(profile_lons) - 0.1, max(profile_lons) + 0.1, 100)  # Extended range
    transect_lats = slope * transect_lons + intercept


    # finds the intersection of two line segments p1p2 and q1q2
    for i in range(len(transect_lons) - 1):
        # Get the transect segment
        p1 = (transect_lons[i], transect_lats[i])
        p2 = (transect_lons[i+1], transect_lats[i+1])

        # Loop over each coastline segment
        for j in range(len(coast_lon) - 1):
            # Get the coastline segment
            q1 = (coast_lon[j], coast_lat[j])
            q2 = (coast_lon[j+1], coast_lat[j+1])

            # Check if the transect segment intersects with the coastline segment
            if do_lines_intersect(p1, p2, q1, q2):
                print(f"Intersection found between transect segment {i} and coastline segment {j}")

                transect_p1 = (transect_lons[i], transect_lats[i])
                transect_p2 = (transect_lons[i+1], transect_lats[i+1])
                coast_q1 = (coast_lon[j], coast_lat[j])
                coast_q2 = (coast_lon[j+1], coast_lat[j+1])
                
                intersect_lon, intersect_lat = line_intersection(transect_p1, transect_p2, coast_q1, coast_q2)
                print(f"Intersection at: {(intersect_lon, intersect_lat)}")


    # --- Step 1. Build transect direction vector in km ---
    ref1 = (transect_lats[0], transect_lons[0])
    ref2 = (transect_lats[-1], transect_lons[-1])

    dx = haversine_np((ref1[0], ref1[1]), (ref1[0], ref2[1]))
    if ref2[1] < ref1[1]:
        dx *= -1

    dy = haversine_np((ref1[0], ref1[1]), (ref2[0], ref1[1]))
    if ref2[0] < ref1[0]:
        dy *= -1

    transect_vec = np.array([dx, dy])
    transect_vec = transect_vec / np.linalg.norm(transect_vec)

    # --- Step 2. Compute along-transect distance in km for each profile ---
    original_distance = []
    for lat, lon in zip(profile_lats, profile_lons):
        dlon = haversine_np((intersect_lat, intersect_lon), (intersect_lat, lon))
        if lon < intersect_lon:
            dlon *= -1

        dlat = haversine_np((intersect_lat, intersect_lon), (lat, intersect_lon))
        if lat < intersect_lat:
            dlat *= -1

        vec = np.array([dlon, dlat])
        original_distance.append(np.dot(vec, transect_vec))

    original_distance = np.array(original_distance)

    # --- Step 3. Sort profiles ---
    sort_idx = np.argsort(original_distance)
    profile_lats   = profile_lats[sort_idx]
    profile_lons   = profile_lons[sort_idx]
    distance = original_distance[sort_idx]

    # --- Step 4. Safeguard: make sure first profile is the most inshore one ---
    # Check which end is closer to the coastline intersection (0 km)
    if abs(distance[-1]) < abs(distance[0]):
        profile_lats = profile_lats[::-1]
        profile_lons = profile_lons[::-1]
        distance = distance[::-1]


    if gebco_bathy:

        df = xr.open_dataset('GIS_data/gebco_2023_n12.0_s-12.0_w-21.0_e16.0.nc')
        
        df = df.sel(lon = slice(region[0], region[1]), lat=slice(region[2], region[3]))
        b_lats = df["lat"].values
        b_lons = df["lon"].values
        d_depths = df["elevation"].values

        b_lons, b_lats = np.meshgrid(b_lons, b_lats)
        
        # Flatten all three
        b_lons = b_lons.ravel()
        b_lats = b_lats.ravel()
        b_depths = d_depths.ravel()
        
    else:
        bathy_file = '/vortexfs1/share/mahadevanlab/datasets/OMI_pilot_campaign_jan2025/bathymetry_data.csv'
        df = pd.read_csv(bathy_file, skiprows=1, names=["time","lat", "lon", "depth","temp"])
        
        b_lats = df["lat"].values
        b_lons = df["lon"].values
        b_depths = df["depth"].values
        
        # remove nans
        valid_mask = ~np.isnan(b_depths)
        b_lons, b_lats, b_depths = b_lons[valid_mask], b_lats[valid_mask], b_depths[valid_mask]
        
    transect_depths = griddata(
        (b_lons, b_lats), b_depths, (profile_lons, profile_lats), method='nearest'
    )

    # filter out points off the shelf
    depth_changes = np.abs(np.diff(transect_depths))

    jump_index = np.where(depth_changes > 150)[0]

    if jump_index.size > 0: 
        first_jump = jump_index[0] + 1  # +1 to include the first invalid point
        transect_depths[first_jump:] = np.nan

    coords = [coords[i] for i in sort_idx]
    datasets = [datasets[i] for i in sort_idx]

    return {
        "profile_lats": profile_lats,
        "profile_lons": profile_lons,
        "distance": distance,
        "transect_depths": transect_depths,
        "transect_lons": transect_lons,
        "transect_lats": transect_lats,
        "intersect_lon": intersect_lon,
        "intersect_lat": intersect_lat,
        "coords": coords,
        "datasets": datasets,
    }

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


def coastline_extraction(lonmin, lonmax, latmin, latmax, coast_file):
    """
    This extracts all coastline in a given domain. Double-check output to remove unneeded coastline 
    (i.e. islands etc).
    """
    coast = gpd.read_file(coast_file)
    np1 = (lonmin, latmax)
    np2 = (lonmax, latmax)
    np3 = (lonmax, latmin)
    np4 = (lonmin, latmin)
    poly = Polygon([np1, np2, np3, np4])
    poly_df = gpd.GeoDataFrame(gpd.GeoSeries(poly), columns=['geometry'], crs='EPSG:4326')
    extract_coast = gpd.overlay(coast, poly_df, how='intersection')
    return extract_coast
    

def extract_polygon_coords(geom, min_area_threshold):
    lon_list = []  # List for longitudes
    lat_list = []  # List for latitudes
    
    if isinstance(geom, MultiPolygon):
        # Iterate through each polygon in the MultiPolygon using .geoms
        for polygon in geom.geoms:
            if polygon.area >= min_area_threshold:
                # Get longitudes (x) and latitudes (y)
                lon_list.extend(polygon.exterior.coords.xy[0])
                lat_list.extend(polygon.exterior.coords.xy[1])
    elif isinstance(geom, Polygon):
        # If it's a simple Polygon, directly check the area and return coords if it passes
        if geom.area >= min_area_threshold:
            lon_list.extend(geom.exterior.coords.xy[0])
            lat_list.extend(geom.exterior.coords.xy[1])

    return lon_list, lat_list  # Return the longitudes and latitudes as separate lists


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


def fill_nan_2d(grid):
    x = np.arange(grid.shape[1])
    y = np.arange(grid.shape[0])
    X, Y = np.meshgrid(x, y)
    valid_mask = ~np.isnan(grid)
    if np.count_nonzero(valid_mask) < 3:
        return grid  # Not enough points to interpolate

    interp_func = griddata(
        (X[valid_mask], Y[valid_mask]),
        grid[valid_mask],
        (X, Y),
        method='linear',
        fill_value=np.nan
    )
    return interp_func