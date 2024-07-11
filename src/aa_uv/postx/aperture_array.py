"""aperture_array: Basic antenna array geometry class."""
import numpy as np
from aa_uv.datamodel import UVX, UVXAntennaCal
from aa_uv.vis_utils import vis_arr_to_matrix_4pol
from astropy.units import Quantity
from loguru import logger
from pygdsm import init_observer

from .aa_plotting import AaPlotter
from .aa_viewer import AllSkyViewer
from .calibration.aa_calibration import AaCalibrator
from .coords.aa_coords import AaCoords
from .imaging.aa_imaging import AaImager
from .simulation.aa_simulation import AaSimulator


class ApertureArray(object):
    """RadioArray class, designed for post-correlation beamforming and all-sky imaging."""

    def __init__(self, uvx: UVX, conjugate_data: bool=False, verbose: bool=False, gsm: str='gsm08'):
        """Initialize RadioArray class (based on astropy EarthLocation).

        Args:
            uvx (UVX):                datamodel visibility dataclass, UVX
            conjugate_data (bool):   Flag to conjugate data (in case upper/lower triangle confusion)
            verbose (bool):          Print extra details to screen
            gsm (str):               Name of global sky model (gsm08, gsm16, lfsm, haslam)
        """
        self.uvx = uvx
        self.conjugate_data = conjugate_data
        self.verbose = verbose
        self.name = uvx.name

        self.earthloc = uvx.origin

        xyz0 = uvx.antennas.attrs['array_origin_geocentric'].data
        self.xyz_enu  = uvx.antennas.enu.data
        self.xyz_ecef = uvx.antennas.ecef.data + xyz0

        self.n_ant = len(self.xyz_enu)
        self.ant_names = uvx.antennas.identifier

        # Setup frequency, time, and phase centre
        self.f = Quantity(uvx.data.coords['frequency'].data, 'Hz')
        self.t = uvx.timestamps
        self.p = uvx.data.polarization.values

        # Phase center
        self.phase_center = uvx.phase_center

        # Setup current index dict and workspace
        self.idx = {'t': 0, 'f': 0, 'p': 0}
        self.workspace = {}

        self.bl_matrix = self._generate_bl_matrix()

        # Healpix workspace
        self._to_workspace('hpx', {})

        # Add-on modules
        self.coords       = AaCoords(self)
        self.plotting     = AaPlotter(self)
        self.viewer       = AllSkyViewer(self)
        self.simulation   = AaSimulator(self)
        self.calibration  = AaCalibrator(self)
        self.imaging      = AaImager(self)
        #self.holography   = AaHolographer(self)

        # Setup Global Sky Model
        self.set_gsm(gsm)

    def __repr__(self):
        """Display representation of ApertureArray object."""
        eloc = self.earthloc.to_geodetic()
        s = f"<ApertureArray: {self.name} (lat {eloc.lat.value:.2f}, lon {eloc.lon.value:.2f})>"
        return s

    def set_gsm(self, gsm_str: str='gsm08'):
        """Set the Global Sky Model to use."""
        self.simulation.model.gsm       = init_observer(gsm_str)
        self.simulation.model.gsm.lat   = self.uvx.origin.lat.to('rad').value
        self.simulation.model.gsm.lon   = self.uvx.origin.lon.to('rad').value
        self.simulation.model.gsm.elev  = self.uvx.origin.height.to('m').value
        self.simulation.model.gsm.date  = self.t[0].datetime
        self.gsm = self.simulation.model.gsm

    def set_idx(self, f: int=None, t: int=None, p: int=None):
        """Set index of UVX data array.

        Args:
            f (int): Frequency index
            t (int): Time index
            p (int): Polarization index

        Notes:
            Updates the self.idx dictionary (keys are f,t,p).
            This controls which data are selected by generate_vis_matrix()
        """
        if f is not None:
            self.idx['f'] = f
        if t is not None:
            self.idx['t'] = t
        if p is not None:
            self.idx['p'] = p

    def _ws(self, key: str):
        """Return value of current index for freq / pol / time or workspace entry.

        Helper function to act as 'workspace'.
        Uses self.idx dictionary which stores selected index

        Args:
            key (str): One of f (freq), p (pol) or t (time)

        Returns:
            Value of f/p/t array at current index in workspace

        """
        if key in ('f', 'p', 't'):
            return self.__getattribute__(key)[self.idx[key]]
        else:
            return self.workspace[key]

    def _in_workspace(self, key: str) -> bool:
        # Check if key is in workspace
        return key in self.workspace

    def _to_workspace(self, key: str, val):
        self.workspace[key] = val


    def _generate_bl_matrix(self):
        """Compute a matrix of baseline lengths."""

        # Helper fn to compute length for one row
        def bl_len(xyz, idx):
            return np.sqrt(np.sum((xyz - xyz[idx])**2, axis=-1))

        bls = np.zeros((self.n_ant, self.n_ant), dtype='float32')
        # Loop over rows
        for ii in range(256):
            bls[ii] = bl_len(self.xyz_enu, ii)
        return bls

    def set_cal(self, cal: UVXAntennaCal):
        """Set gaincal solution (used when generating vis matrix).

        Args:
            cal (UVXAntennaCal): Calibration to apply
        """
        self.workspace['cal'] = cal

    def generate_vis_matrix(self, vis: str='data', t_idx=None, f_idx=None) -> np.array:
        """Generate visibility matrix from underlying array data.

        Underlying UVX data has axes (time, frequency, baseline, polarization)
        Model data should have axes (time, frequency, antenna1, antenna2) and be an xr.DataArray

        Args:
            vis (str): Select visibilities to be either raw 'data', calibrated 'corrected', or 'model'
            t_idx (int): Time index for data
            f_idx (int): Frequency index for data

        Returns:
            vis_mat (np.array): Visibility data
        """
        t_idx = self.idx['t'] if t_idx is None else t_idx
        f_idx = self.idx['f'] if f_idx is None else f_idx
        match vis:
            case 'corrected':
                vis_mat = self.workspace['vis']['corrected']
            case 'data':
                vis_sel = self.uvx.data[t_idx, f_idx].values
                vis_mat = vis_arr_to_matrix_4pol(vis_sel, self.n_ant)
            case 'cal':
                vis_sel = self.uvx.data[t_idx, f_idx].values
                vis_mat = vis_arr_to_matrix_4pol(vis_sel, self.n_ant)
                if 'cal' in self.workspace.keys():
                    vis_mat *= self.workspace['cal'].to_matrix()
                else:
                    logger.warning("Calibration not set, returning raw visibilities.")
            case 'model':
                vis_mat = self.simulation.model.visibilities[t_idx, f_idx].values

        return vis_mat
