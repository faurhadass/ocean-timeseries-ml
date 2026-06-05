def load_ocean_data(filename, nz, nx, nt, latitude, start_time, time_resolution=1):
    """Direct translation of the MATLAB load_ocean_data.

    File columns: [lon, depth, temp, salinity, u, v].
    Returns a dict with 3D fields indexed [depth, lon, time] (like a struct).
    """
    raw = pd.read_csv(filename).values  # [nx*nz*nt, 6]

    if raw.shape[0] != nx * nz * nt:
        raise ValueError("Data length does not match nx*nz*nt. Check input values.")

    temperature = np.zeros((nz, nx, nt))
    salinity    = np.zeros((nz, nx, nt))
    u           = np.zeros((nz, nx, nt))
    v           = np.zeros((nz, nx, nt))
    depth       = np.zeros((nz, nx))
    lon         = np.zeros((nz, nx))

    for t in range(nt):
        frame = raw[t*nx*nz:(t+1)*nx*nz, :]            # [nz*nx, 6]
        # MATLAB reshape is column-major -> order="F"
        temperature[:, :, t] = frame[:, 2].reshape(nz, nx, order="F")
        salinity[:, :, t]    = frame[:, 3].reshape(nz, nx, order="F")
        u[:, :, t]           = frame[:, 4].reshape(nz, nx, order="F")
        v[:, :, t]           = frame[:, 5].reshape(nz, nx, order="F")
        if t == 0:
            depth = frame[:, 1].reshape(nz, nx, order="F")
            lon   = frame[:, 0].reshape(nz, nx, order="F")

    # Generate time vector
    time_vec = pd.date_range(start=start_time, periods=nt, freq=f"{time_resolution}h")

    # --- Adjust for leap years across multi-year datasets ---
    if time_resolution == 1:  # only valid for hourly data
        # Find all Feb 29ths at 00:00 in the time vector
        is_feb29 = (time_vec.month == 2) & (time_vec.day == 29) & (time_vec.hour == 0)
        idx_feb29 = np.where(is_feb29)[0]

        remove = np.zeros(len(time_vec), dtype=bool)
        for idx in idx_feb29:
            end = idx + 24                 # this hour plus the next 23
            if end <= len(time_vec):
                remove[idx:end] = True

        if remove.any():
            keep = ~remove
            temperature = temperature[:, :, keep]
            salinity    = salinity[:, :, keep]
            u           = u[:, :, keep]
            v           = v[:, :, keep]
            time_vec    = time_vec[keep]
            nt          = temperature.shape[2]   # update nt after removal

    return {
        "temp": temperature, "sal": salinity, "u": u, "v": v,
        "depth": depth, "lon": lon, "time": time_vec,
        "lat": latitude, "nx": nx, "nz": nz, "nt": nt,
        "timeRes": time_resolution,
    }