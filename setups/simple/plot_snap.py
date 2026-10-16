"""
plot_snap: quick look at the snapshot file of a script configuration

    python plot_snap.py acc1.cdf [time index, default -1]

Surface fields, zonal mean sections and the kinetic energy time series.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

name = sys.argv[1]
it   = int(sys.argv[2]) if len(sys.argv) > 2 else -1
f    = Dataset(name)
t    = f.variables['time'][:]
x, y, z = f.variables['xt'][:], f.variables['yt'][:], f.variables['zt'][:]
u, v, temp, p_s = (np.ma.masked_invalid(f.variables[n][it]) for n in ('u', 'v', 'temp', 'p_s'))
ks = -1                                                     # surface level
xunit = 'deg' if abs(x).max() < 400 else 'km'
xs = 1. if xunit == 'deg' else 1e-3

fig, ax = plt.subplots(2, 3, figsize=(14, 8))
for a, field, label in zip(ax[0], (temp[ks], u[ks], p_s), ('surface temp / buoyancy', 'surface u [m/s]', 'p_s [m^2/s^2]')):
    c = a.pcolormesh(x*xs, y*xs, field, cmap='RdBu_r'); plt.colorbar(c, ax=a); a.set_title(label)
    a.set_xlabel('x [%s]' % xunit); a.set_ylabel('y [%s]' % xunit)
for a, field, label in zip(ax[1][:2], (temp.mean(axis=2), u.mean(axis=2)), ('zonal mean temp / buoyancy', 'zonal mean u [m/s]')):
    c = a.pcolormesh(y*xs, z, field, cmap='RdBu_r'); plt.colorbar(c, ax=a); a.set_title(label)
    a.set_xlabel('y [%s]' % xunit); a.set_ylabel('z [m]')
ke = [0.5*np.nanmean(f.variables['u'][i]**2 + f.variables['v'][i]**2) for i in range(len(t))]
ax[1][2].plot(t/86400., ke); ax[1][2].set_xlabel('t [days]'); ax[1][2].set_title('mean kinetic energy [m^2/s^2]')
fig.suptitle('%s, t = %.1f days' % (name, t[it]/86400.))
plt.tight_layout()
plt.show()
