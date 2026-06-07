import numpy as np
import pandas as pd


def load_ocean_data(filename, nz, nx, nt, latitude, start_time, time_resolution=1):
    """Load a latitude cross-section of oceanographic data from POM text output.

    The input file contains one record per line with six space-separated columns:
    longitude (degrees E), depth (m), temperature (degrees C), salinity (psu),
    eastward velocity u (m/s), and northward velocity v (m/s). Records are ordered
    with depth (Z) varying fastest, then longitude (X), then time, so the file holds
    nt consecutive blocks of (nz * nx) rows, one block per time step.

    Parameters
    ----------
    filename : str
        Path to the cross-section data file.
    nz : int
        Number of depth levels.
    nx : int
        Number of longitude points.
    nt : int
        Number of time steps.
    latitude : float
        Latitude of the cross-section (constant for the whole file).
    start_time : pandas.Timestamp or datetime
        Timestamp of the first time step.
    time_resolution : int, optional
        Hours between time steps (default 1, i.e. hourly data).

    Returns
    -------
    dict
        temp, sal, u, v : ndarray, shape [nz, nx, nt]
            3D fields indexed [depth, longitude, time].
        depth : ndarray, shape [nz, nx]
            Depth of each grid cell. 2D because vertical levels sit at different
            actual depths at each longitude (the grid is non-uniform, with finer
            spacing closer to shore).
        lon : ndarray, shape [nx]
            Longitude values. 1D because longitude is constant down each column.
        time : pandas.DatetimeIndex, length nt
            Timestamp of each time step.
        lat : float
            Latitude of the cross-section.
        nx, nz, nt : int
            Grid dimensions.
        timeRes : int
            Time resolution in hours.
    """
    # Read the space-delimited file (no header row) into a 2D array of shape [nz*nx*nt, 6]
    raw = pd.read_csv(filename, sep=r"\s+", header=None).values

    # Verify the file length matches the expected number of records
    if raw.shape[0] != nx * nz * nt:
        raise ValueError("Data length does not match nx*nz*nt. Check input values.")

    # Preallocate output arrays
    temperature = np.zeros((nz, nx, nt))   # [depth, lon, time]
    salinity    = np.zeros((nz, nx, nt))   # [depth, lon, time]
    u           = np.zeros((nz, nx, nt))   # [depth, lon, time]
    v           = np.zeros((nz, nx, nt))   # [depth, lon, time]
    depth       = np.zeros((nz, nx))       # [depth, lon] — varies along both axes
    lon         = np.zeros(nx)             # [lon]        — constant with depth

    # Fill arrays one time step at a time
    for t in range(nt):
        frame = raw[t*nx*nz:(t+1)*nx*nz, :]   # rows for time step t, shape [nz*nx, 6]

        # Reshape each column to [depth, lon]. order="F" (column-major) matches the
        # file's depth-fastest ordering, equivalent to MATLAB's default reshape.
        temperature[:, :, t] = frame[:, 2].reshape(nz, nx, order="F")
        salinity[:, :, t]    = frame[:, 3].reshape(nz, nx, order="F")
        u[:, :, t]           = frame[:, 4].reshape(nz, nx, order="F")
        v[:, :, t]           = frame[:, 5].reshape(nz, nx, order="F")

        # Depth and longitude are constant in time, so capture them once
        if t == 0:
            depth = frame[:, 1].reshape(nz, nx, order="F")          # keep 2D
            lon   = frame[:, 0].reshape(nz, nx, order="F")[0, :]    # one row -> 1D

    # Build the time axis: nt evenly spaced timestamps from start_time
    time_vec = pd.date_range(start=start_time, periods=nt, freq=f"{time_resolution}h")

    return {
        "temp": temperature, "sal": salinity, "u": u, "v": v,
        "depth": depth, "lon": lon, "time": time_vec,
        "lat": latitude, "nx": nx, "nz": nz, "nt": nt,
        "timeRes": time_resolution,
    }