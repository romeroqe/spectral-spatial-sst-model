import numpy as np
import xarray as xr

def predictor_in_pixel(lat, lon, ds_modes, t, t0=0):
    is_spatial = 'lat' in ds_modes.coords and 'lon' in ds_modes.coords

    if is_spatial:
        ilat = np.abs(ds_modes.lat - lat).argmin().item()
        ilon = np.abs(ds_modes.lon - lon).argmin().item()

        n_modes = ds_modes.n_modes.isel(lat=ilat, lon=ilon).item()
        periods = ds_modes.period[:n_modes, ilat, ilon].values
        amplitudes = ds_modes.amplitude[:n_modes, ilat, ilon].values
        phases = ds_modes.phase[:n_modes, ilat, ilon].values
    else:
        n_modes = ds_modes.n_modes.item()
        periods = ds_modes.period[:n_modes].values
        amplitudes = ds_modes.amplitude[:n_modes].values
        phases = ds_modes.phase[:n_modes].values

    return reconstruct_signal(t, periods, amplitudes, phases, t0=t0)

def reconstruct_signal(t, periods, amplitudes, phases, t0=0):
    signal = np.zeros_like(t, dtype=float)
    for T, A, phi in zip(periods, amplitudes, phases):
        signal += A * np.cos(2*np.pi*((t - t0)/T) + phi)
    return signal

def reconstruct_predictors(gam_predictors_ds, predictors_modes_dict, time):
    lat_len = gam_predictors_ds.lat.size
    lon_len = gam_predictors_ds.lon.size
    time_len = time.size
    pred_names = list(gam_predictors_ds.data_vars)
    n_pred = len(pred_names)
    recon = np.zeros((time_len, n_pred, lat_len, lon_len), dtype=float)
    
    for i in range(lat_len):
        for j in range(lon_len):
            lat_val = gam_predictors_ds.lat[i].item()
            lon_val = gam_predictors_ds.lon[j].item()
            
            for k, predictor_name in enumerate(pred_names):
                if gam_predictors_ds[predictor_name].isel(lat=i, lon=j).item() == 1:
                    ds_modes = predictors_modes_dict[predictor_name]

                    t = np.arange(len(time))
                    ref = ds_modes.attrs["time_reference"].replace("t0 =", "").strip()
                    ref = np.datetime64(ref, "M")
                    time0 = time[0].astype("datetime64[M]")
                    t0 = (ref - time0).astype(int)

                    recon[:, k, i, j] = predictor_in_pixel(lat_val, lon_val, ds_modes, t, t0)

    da_recon = xr.DataArray(
        recon,
        dims=("time", "predictor", "lat", "lon"),
        coords={"time": t, "predictor": pred_names, "lat": gam_predictors_ds.lat, "lon": gam_predictors_ds.lon},
        name="predictor_signal"
    )
    return da_recon.to_dataset()