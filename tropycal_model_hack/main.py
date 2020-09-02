import os
import glob
import pickle
import argparse

import matplotlib
matplotlib.use('agg')

import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import tropycal.tracks as tracks


COLUMNS = ['basin', 'number', 'init', 'technum', 'tech', 'tau',
           'lat', 'lon', 'vmax', 'mslp', 'level', 'rad', 'windcode', 'rad1',
           'rad2', 'rad3', 'rad4', 'radp', 'rrp', 'mrd', 'gusts', 'eye',
           'subregion', 'maxseas', 'initials', 'dir', 'speed', 'stormname',
           'depth', 'seas', 'seascode', 'seas1', 'seas2', 'seas3', 'seas4']
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
CACHED_FP = os.path.join(DATA_DIR, 'storm.pkl')


def _scale_lat_lon(df, col, endswith):
    idx = df[col].astype(str).str.endswith(endswith)
    if sum(idx) == 0:  # all False (no matches)
        return df

    scaler = -0.1 if endswith in ['W', 'S'] else 0.1
    df.loc[idx, col] = (
        df.loc[idx, col].str[:-1]  # strip the one letter (N/S/E/W)
        .astype(float) * scaler)
    return df


def plot_global_tracks(track_filenames, output_image):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    for in_fp in track_filenames:
        # read local data, rename, and create cols
        df = pd.read_csv(in_fp, header=None, names=COLUMNS)
        df = df.drop_duplicates(['init', 'tech', 'tau'], keep='first')
        latlon_dirs = [('lat', 'N'), ('lat', 'S'),
                       ('lon', 'E'), ('lon', 'W')]
        for latlon, endswith in latlon_dirs:
            df = _scale_lat_lon(df, latlon, endswith)
        for tech, df_group in df.groupby('tech'):
            if tech == ' C00Z':
                _ = ax.plot(df_group['lon'], df_group['lat'], c='whitesmoke',
                            transform=ccrs.PlateCarree(), zorder=10)
                df_subset = df_group.loc[df_group['vmax'] > 64]
                if len(df_subset):
                    _ = ax.plot(
                        df_subset['lon'], df_subset['lat'], c='red',
                        transform=ccrs.PlateCarree(), zorder=20)
            else:
                _ = ax.plot(df_group['lon'], df_group['lat'], c='gray',
                            transform=ccrs.PlateCarree())
    ax.set_ylim(-60, 60)
    ax.background_img(name='BM', resolution='high')
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0)


def plot_forecast_cone(in_fp, out_fp, storm_label=None):
    """
    Repopulate tropycal storm dict with local model data; then
    invoke plot_nhc_forecast and outputs a TC track cone forecast figure.

    Args:
        in_fp (str) - input filepath
        out_fp (str) - output filepath
        storm_label (str) - storm label; if not provided, use default tech
    """
    try:
        with open(CACHED_FP, 'rb') as f:
            storm = pickle.load(f)
    except Exception as e:
        print(e)
        # read nhc data for prepopulation
        hurdat_atl = tracks.TrackDataset(
            basin='north_atlantic', source='hurdat',
            include_btk=False)
        storm = hurdat_atl.get_storm(('michael', 2018))
        with open(CACHED_FP, 'wb') as f:
            pickle.dump(storm, f)

    # read local data, rename, and create cols
    df = pd.read_csv(in_fp, header=None, names=COLUMNS)
    df = df.loc[df['tech'] == ' C00Z']
    df = df.drop_duplicates(['tau'], keep='first')
    df['ace'] = df['vmax'] ** 2 / 10000
    df['date'] = (
        pd.to_datetime(df['init'], format='%Y%m%d%H') +
        pd.to_timedelta(df['tau'], unit='H'))
    df['season'] = df['date'].dt.year
    df['year'] = df['date'].dt.year
    df['source'] = 'model'
    df['source_info'] = 'FNMOC'
    df['special'] = ''
    df['type'] = 'TS'
    df['extra_obs'] = 0
    df['wmo_basin'] = df['basin']
    latlon_dirs = [('lat', 'N'), ('lat', 'S'),
                   ('lon', 'E'), ('lon', 'W')]
    for latlon, endswith in latlon_dirs:
        df = _scale_lat_lon(df, latlon, endswith)
    df = df.rename(columns={
        'stormname': 'name',
        'tech': 'id',
        'technum': 'operational_id',
        'tau': 'fhr'
    })
    if storm_label is not None:
        df['name'] = storm_label
    else:
        df['name'] = df['id']
    print(storm.dict)

    # replace values
    for key in storm.dict:
        if isinstance(storm.dict[key], (str, int, float)):
            storm.dict[key] = df[key].unique()[0]
        elif key == 'date':
            storm.dict[key] = [pd_dt.to_pydatetime() for pd_dt in df[key]]
        elif isinstance(storm.dict[key], list):
            storm.dict[key] = df[key].values.tolist()
        else:
            storm.dict[key] = df[key].values

    sub_df = df[['init', 'fhr', 'lat', 'lon', 'vmax', 'mslp', 'type']]
    storm.forecast_dict = {}
    # not really official forecast
    # tropycal only uses OFCL for cone forecast
    storm.forecast_dict['OFCL'] = {}
    for init, group in sub_df.groupby('init'):
        storm.forecast_dict['OFCL'][init] = (
            group.drop(columns='init').to_dict('list'))
        storm.forecast_dict['OFCL'][init]['init'] = pd.to_datetime(
            init, format='%Y%m%d%H').to_pydatetime()

    # to use plot_nhc_forecast CARQ is expected
    storm.forecast_dict['CARQ'] = storm.forecast_dict['OFCL']
    
    try:
        ax = storm.plot_nhc_forecast(0, return_ax=True)
    except Exception as e:
        ax = storm.plot(return_ax=True)
    plt.savefig(out_fp)


def plot_forecast_cone_cli():
    parser = argparse.ArgumentParser(
        description="""
            Output a tropical cyclone forecast cone figure.
        """
    )
    parser.add_argument('in_fp', type=str, help='input filepath')
    parser.add_argument('out_fp', type=str, help='output filepath')
    parser.add_argument('storm_label', type=str, help='label of storm',
                        nargs='?', const=None, default=None)
    kwargs = vars(parser.parse_args())
    plot_forecast_cone(**kwargs)


def plot_global_tracks_cli():
    parser = argparse.ArgumentParser(
        description="""
            Output a global tropical cyclone forecast figure.
        """
    )
    parser.add_argument('-o', '--output_image', type=str, help='output image')
    parser.add_argument('track_filenames', nargs='+',
                        help='track filenames')
    kwargs = vars(parser.parse_args())
    plot_global_tracks(**kwargs)
