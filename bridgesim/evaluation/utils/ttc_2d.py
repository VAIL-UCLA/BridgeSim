########################################################################################################
#
# Public API:
#   compute_ttc_2d(samples, 'dataframe'|'values')
#       2D Time-To-Collision using rectangular bounding-box geometry (Jiao 2022).
#       Constant-velocity model. Returns np.inf (no collision) or negative (boxes overlap).
#
#   compute_mttc_2d(samples, 'dataframe'|'values')
#       2D Modified TTC accounting for longitudinal acceleration (requires column 'acc_i').
#       Same bounding-box geometry as compute_ttc_2d. Falls back to TTC when |Δa| < 1e-6.
#
#   compute_mttc_1d(samples, 'dataframe'|'values')
#       1D Modified TTC: projects relative velocity/acceleration onto ego→agent centerline.
#       Simpler approximation for comparison with compute_mttc_2d.
#
#   compute_current_dist_2d(samples, 'dataframe'|'values')
#       Current minimum distance between bounding boxes (0 if overlapping).
#
# Required input columns (pandas DataFrame):
#   x_i, y_i, vx_i, vy_i, hx_i, hy_i, length_i, width_i  (ego vehicle)
#   x_j, y_j, vx_j, vy_j, hx_j, hy_j, length_j, width_j  (other vehicle)
#   acc_i  [optional, required for MTTC functions]
#   acc_j  [optional, defaults to 0 for MTTC functions]
#
########################## Copyright (c) 2022 Yiru Jiao <y.jiao-1@tudelft.nl> ###########################

import numpy as np
import pandas as pd
import warnings


def line(point0, point1):
    x0, y0 = point0
    x1, y1 = point1
    a = y0 - y1
    b = x1 - x0
    c = x0*y1 - x1*y0
    return a, b, c

def intersect(line0, line1):
    a0, b0, c0 = line0
    a1, b1, c1 = line1
    D = a0*b1 - a1*b0 # D==0 then two lines overlap
    D[D==0] = np.nan
    x = (b0*c1 - b1*c0)/D
    y = (a1*c0 - a0*c1)/D
    return np.array([x, y])

def ison(line_start, line_end, point):
    crossproduct = (point[1]-line_start[1])*(line_end[0]-line_start[0]) - (point[0]-line_start[0])*(line_end[1]-line_start[1])
    dotproduct = (point[0]-line_start[0])*(line_end[0]-line_start[0]) + (point[1]-line_start[1])*(line_end[1]-line_start[1])
    squaredlength = (line_end[0]-line_start[0])**2 + (line_end[1]-line_start[1])**2
    return (np.absolute(crossproduct)<=1e-5)&(dotproduct>=0)&(dotproduct<=squaredlength)

def dist_p2l(point, line_start, line_end):
    return np.absolute((line_end[0]-line_start[0])*(line_start[1]-point[1])-(line_start[0]-point[0])*(line_end[1]-line_start[1]))/np.sqrt((line_end[0]-line_start[0])**2+(line_end[1]-line_start[1])**2)

def getpoints(samples):
    ## vehicle i
    heading_i = samples[['hx_i','hy_i']].values
    perp_heading_i = np.array([-heading_i[:,1], heading_i[:,0]]).T
    heading_scale_i = np.tile(np.sqrt(heading_i[:,0]**2+heading_i[:,1]**2), (2,1)).T
    length_i = np.tile(samples.length_i.values, (2,1)).T
    width_i = np.tile(samples.width_i.values, (2,1)).T

    point_up = samples[['x_i','y_i']].values + heading_i/heading_scale_i*length_i/2
    point_down = samples[['x_i','y_i']].values - heading_i/heading_scale_i*length_i/2
    point_i1 = (point_up + perp_heading_i/heading_scale_i*width_i/2).T
    point_i2 = (point_up - perp_heading_i/heading_scale_i*width_i/2).T
    point_i3 = (point_down + perp_heading_i/heading_scale_i*width_i/2).T
    point_i4 = (point_down - perp_heading_i/heading_scale_i*width_i/2).T

    ## vehicle j
    heading_j = samples[['hx_j','hy_j']].values
    perp_heading_j = np.array([-heading_j[:,1], heading_j[:,0]]).T
    heading_scale_j= np.tile(np.sqrt(heading_j[:,0]**2+heading_j[:,1]**2), (2,1)).T
    length_j = np.tile(samples.length_j.values, (2,1)).T
    width_j = np.tile(samples.width_j.values, (2,1)).T

    point_up = samples[['x_j','y_j']].values + heading_j/heading_scale_j*length_j/2
    point_down = samples[['x_j','y_j']].values - heading_j/heading_scale_j*length_j/2
    point_j1 = (point_up + perp_heading_j/heading_scale_j*width_j/2).T
    point_j2 = (point_up - perp_heading_j/heading_scale_j*width_j/2).T
    point_j3 = (point_down + perp_heading_j/heading_scale_j*width_j/2).T
    point_j4 = (point_down - perp_heading_j/heading_scale_j*width_j/2).T

    return (point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4)

def compute_current_dist_2d(samples, toreturn='dataframe'):
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)

        dist_mat = []
        count_i = 0
        for point_i_start, point_i_end in zip([point_i1, point_i4, point_i3, point_i2],[point_i2, point_i3, point_i1, point_i4]):
            count_j = 0
            for point_j_start, point_j_end in zip([point_j1, point_j4, point_j3, point_j2],[point_j2, point_j3, point_j1, point_j4]):
                if count_i<2 and count_j<2 :
                    # Distance from point to point
                    dist_mat.append(np.sqrt((point_i_start[0]-point_j_start[0])**2+(point_i_start[1]-point_j_start[1])**2))
                    dist_mat.append(np.sqrt((point_i_start[0]-point_j_end[0])**2+(point_i_start[1]-point_j_end[1])**2))
                    dist_mat.append(np.sqrt((point_i_end[0]-point_j_start[0])**2+(point_i_end[1]-point_j_start[1])**2))
                    dist_mat.append(np.sqrt((point_i_end[0]-point_j_end[0])**2+(point_i_end[1]-point_j_end[1])**2))

                # Distance from point to edge
                ist = intersect(line(point_i_start, point_i_start+np.array([-(point_j_start-point_j_end)[1],(point_j_start-point_j_end)[0]])), line(point_j_start, point_j_end))
                ist[:,~ison(point_j_start, point_j_end, ist)] = np.nan
                dist_mat.append(np.sqrt((ist[0]-point_i_start[0])**2+(ist[1]-point_i_start[1])**2))

                # Overlapped bounding boxes
                ist = intersect(line(point_i_start, point_i_end), line(point_j_start, point_j_end))
                dist = np.ones(len(samples))*np.nan
                dist[ison(point_i_start, point_i_end, ist)&ison(point_j_start, point_j_end, ist)] = 0
                dist[np.isnan(ist[0])&(ison(point_i_start, point_i_end, point_j_start)|ison(point_i_start, point_i_end, point_j_end))] = 0
                dist_mat.append(dist)
                count_j += 1
            count_i += 1

        cdist = np.nanmin(np.array(dist_mat), axis=0)

        if toreturn=='dataframe':
            samples['CurrentD'] = cdist
            return samples
        elif toreturn=='values':
            return cdist

def _ttc_2d_ij(samples):
    point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)
    direct_v = (samples[['vx_i','vy_i']].values - samples[['vx_j','vy_j']].values).T

    dist_mat = []
    leaving_mat = []
    for point_line_start in [point_i1,point_i2,point_i3,point_i4]:
        for edge_start, edge_end in zip([point_j1, point_j3, point_j1, point_j2],[point_j2, point_j4, point_j3, point_j4]):
            point_line_end = point_line_start+direct_v
            ### intersection point
            ist = intersect(line(point_line_start, point_line_end), line(edge_start, edge_end))
            ist[:,~ison(edge_start, edge_end, ist)] = np.nan
            ### distance from point to intersection point
            dist_ist = np.sqrt((ist[0]-point_line_start[0])**2+(ist[1]-point_line_start[1])**2)
            dist_ist[np.isnan(dist_ist)] = np.inf
            dist_mat.append(dist_ist)
            leaving = direct_v[0]*(ist[0]-point_line_start[0]) + direct_v[1]*(ist[1]-point_line_start[1])
            leaving[leaving>=0] = 10
            leaving[leaving<0] = 1
            leaving_mat.append(leaving)

    dist2overlap = np.array(dist_mat).min(axis=0)
    TTC = dist2overlap/np.sqrt((samples.vx_i-samples.vx_j)**2+(samples.vy_i-samples.vy_j)**2)
    leaving = np.nansum(np.array(leaving_mat),axis=0)
    TTC[leaving<10] = np.inf
    TTC[(leaving>10)&(leaving%10!=0)] = -1

    return TTC


def compute_ttc_2d(samples, toreturn='dataframe'):
    if toreturn!='dataframe' and toreturn!='values':
        warnings.warn('Incorrect target to return. Please specify \'dataframe\' or \'values\'.')
    else:
        ttc_ij = _ttc_2d_ij(samples)
        keys = [var+'_i' for var in ['x','y','vx','vy','hx','hy','length','width']]
        values = [var+'_j' for var in ['x','y','vx','vy','hx','hy','length','width']]
        keys.extend(values)
        values.extend(keys)
        rename_dict = {keys[i]: values[i] for i in range(len(keys))}
        ttc_ji = _ttc_2d_ij(samples.rename(columns=rename_dict))

        if toreturn=='dataframe':
            samples['TTC'] = np.minimum(ttc_ij, ttc_ji)
            return samples
        elif toreturn=='values':
            return np.minimum(ttc_ij, ttc_ji)


def _dtc_2d_ij(samples):
    """Distance-to-collision from vehicle i to j, and leaving indicator (20=approaching, 1=leaving)."""
    point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = getpoints(samples)
    relative_v = (samples[['vx_i','vy_i']].values - samples[['vx_j','vy_j']].values).T

    dist_mat = []
    leaving_mat = []
    for point_line_start in [point_i1, point_i2, point_i3, point_i4]:
        for edge_start, edge_end in zip([point_j1, point_j3, point_j1, point_j2],
                                        [point_j2, point_j4, point_j3, point_j4]):
            point_line_end = point_line_start + relative_v
            ist = intersect(line(point_line_start, point_line_end), line(edge_start, edge_end))
            ist[:, ~ison(edge_start, edge_end, ist)] = np.nan
            dist_ist = np.sqrt((ist[0] - point_line_start[0])**2 + (ist[1] - point_line_start[1])**2)
            dist_ist[np.isnan(dist_ist)] = np.inf
            dist_mat.append(dist_ist)
            leaving = relative_v[0] * (ist[0] - point_line_start[0]) + relative_v[1] * (ist[1] - point_line_start[1])
            leaving[leaving >= 0] = 20
            leaving[leaving < 0] = 1
            leaving_mat.append(leaving)

    dtc = np.array(dist_mat).min(axis=0)
    leaving = np.nansum(np.array(leaving_mat), axis=0)
    return dtc, leaving


def compute_mttc_2d(samples, toreturn='dataframe'):
    """
    Compute 2D Modified Time-To-Collision accounting for acceleration.
    Requires column 'acc_i' (ego longitudinal scalar acceleration).
    Optional column 'acc_j' (other vehicle longitudinal scalar acceleration); defaults to 0.
    Returns np.inf if vehicles will not collide; -1 if bounding boxes overlap.
    Falls back to TTC when |delta_a| < 1e-6.
    """
    if toreturn not in ('dataframe', 'values'):
        warnings.warn("Incorrect target to return. Please specify 'dataframe' or 'values'.")
        return
    if 'acc_i' not in samples.columns:
        warnings.warn("Acceleration of the ego vehicle (acc_i) is not provided.")
        return

    delta_v = np.sqrt((samples['vx_i'] - samples['vx_j'])**2 + (samples['vy_i'] - samples['vy_j'])**2)

    dtc_ij, leaving_ij = _dtc_2d_ij(samples)
    ttc_ij = dtc_ij / delta_v
    ttc_ij[leaving_ij < 20] = np.inf
    ttc_ij[(leaving_ij > 20) & (leaving_ij % 20 != 0)] = -1

    keys = [var+'_i' for var in ['x','y','vx','vy','hx','hy','length','width']]
    values = [var+'_j' for var in ['x','y','vx','vy','hx','hy','length','width']]
    keys.extend(values)
    values.extend(keys)
    rename_dict = {keys[i]: values[i] for i in range(len(keys))}
    dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
    ttc_ji = dtc_ji / delta_v
    ttc_ji[leaving_ji < 20] = np.inf
    ttc_ji[(leaving_ji > 20) & (leaving_ji % 20 != 0)] = -1

    ttc = np.minimum(ttc_ij, ttc_ji)
    dtc = np.minimum(dtc_ij, dtc_ji)

    acc_i = samples['acc_i'].values
    acc_j = samples['acc_j'].values if 'acc_j' in samples.columns else np.zeros(len(samples))
    delta_a = acc_i - acc_j

    # Negative delta_v when vehicles are leaving each other
    delta_v = delta_v * np.sign(((leaving_ij >= 20) | (leaving_ji >= 20)).astype(int) - 0.5)

    squared_term = delta_v**2 + 2 * delta_a * dtc
    sqrt_term = np.where(squared_term >= 0, np.sqrt(np.maximum(squared_term, 0.0)), np.nan)

    mttc_plus  = (-delta_v + sqrt_term) / delta_a
    mttc_minus = (-delta_v - sqrt_term) / delta_a

    mttc = mttc_minus.copy()
    mttc[(mttc_minus <= 0) & (mttc_plus > 0)] = mttc_plus[(mttc_minus <= 0) & (mttc_plus > 0)]
    mttc[(mttc_minus <= 0) & (mttc_plus <= 0)] = np.inf
    mttc[np.isnan(mttc_minus) | np.isnan(mttc_plus)] = np.inf
    mttc[np.abs(delta_a) < 1e-6] = ttc[np.abs(delta_a) < 1e-6]
    mttc[((leaving_ij > 20) & (leaving_ij % 20 != 0)) | ((leaving_ji > 20) & (leaving_ji % 20 != 0))] = -1

    if toreturn == 'dataframe':
        samples['MTTC'] = mttc
        return samples
    else:
        return mttc


def compute_mttc_1d(samples, toreturn='dataframe'):
    """
    1D Modified Time-To-Collision (for comparison with 2D MTTC).
    Projects relative velocity and acceleration onto the ego->agent centerline direction.
    Distance approximated as center distance minus sum of half-diagonals.
    Requires column 'acc_i'. Optional 'acc_j' (defaults to 0).
    Returns np.inf if no collision; -1 if already overlapping.
    """
    if toreturn not in ('dataframe', 'values'):
        warnings.warn("Incorrect target to return. Please specify 'dataframe' or 'values'.")
        return
    if 'acc_i' not in samples.columns:
        warnings.warn("Acceleration of the ego vehicle (acc_i) is not provided.")
        return

    dx = samples['x_j'].values - samples['x_i'].values
    dy = samples['y_j'].values - samples['y_i'].values
    center_dist = np.sqrt(dx**2 + dy**2)

    half_diag_i = np.sqrt((samples['length_i'].values / 2)**2 + (samples['width_i'].values / 2)**2)
    half_diag_j = np.sqrt((samples['length_j'].values / 2)**2 + (samples['width_j'].values / 2)**2)
    d = np.maximum(center_dist - half_diag_i - half_diag_j, 0.0)

    safe_dist = np.where(center_dist > 1e-6, center_dist, 1.0)
    ux = dx / safe_dist
    uy = dy / safe_dist

    v_rel = (samples['vx_i'].values - samples['vx_j'].values) * ux + \
            (samples['vy_i'].values - samples['vy_j'].values) * uy

    acc_i = samples['acc_i'].values
    acc_j = samples['acc_j'].values if 'acc_j' in samples.columns else np.zeros(len(samples))
    a_rel = (acc_i * samples['hx_i'].values - acc_j * samples['hx_j'].values) * ux + \
            (acc_i * samples['hy_i'].values - acc_j * samples['hy_j'].values) * uy

    mttc = np.full(len(samples), np.inf)
    mttc[center_dist < 1e-6] = -1.0

    eps = 1e-6
    linear = np.abs(a_rel) < eps
    with np.errstate(divide='ignore', invalid='ignore'):
        ttc_linear = np.where(v_rel > eps, d / v_rel, np.inf)
    mttc[linear] = ttc_linear[linear]

    quad = ~linear & (center_dist >= 1e-6)
    disc = v_rel**2 + 2 * a_rel * d
    disc_valid = quad & (disc >= 0)
    sqrt_disc = np.where(disc_valid, np.sqrt(np.maximum(disc, 0.0)), np.nan)
    t1 = (-v_rel + sqrt_disc) / a_rel
    t2 = (-v_rel - sqrt_disc) / a_rel
    mttc_quad = np.minimum(np.where(t1 > 0, t1, np.inf), np.where(t2 > 0, t2, np.inf))
    mttc[disc_valid] = mttc_quad[disc_valid]

    if toreturn == 'dataframe':
        samples['MTTC_1D'] = mttc
        return samples
    else:
        return mttc
