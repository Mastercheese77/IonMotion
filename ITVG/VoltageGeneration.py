"""
This module generates electrode voltage sets (i.e. waveforms)
"""

import collections
import csv
import functools
import itertools
import os
import re
import time

import numpy as np
import numpy.linalg as la
import scipy
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as sciopt


ELEM_CHARGE = 1.602e-19  # in coulombs
amu = 1.66e-27  # in kg
EPSILON = 1e-15


def get_spaces(lattice_data):
    """
    Given `lattice_data` an array of shape (N, M)
    where M >= 3 such that lattice_data[:, 0:3]
    is a Cartesian product X x Y x Z,
    return ((X, Y, Z), (len(X), len(Y), len(Z)).
    """
    spaces_list = []
    dimensions_list = []

    for i in range(3):
        coord_set = set(lattice_data[:, i])
        min_val, max_val = min(coord_set), max(coord_set)
        n = len(coord_set)
        space = np.linspace(min_val, max_val, n)

        spaces_list.append(space)
        dimensions_list.append(n)

    return tuple(spaces_list), tuple(dimensions_list)


def compute_pseudopot(e_field_arr, v_rf, mass, rf_omega, scale):
    """
    Compute the pseudopotential at N points given the electric field
    produced by 1 Vdc on the electrode at those same points.

    Parameters
    ----------
    e_field_arr : array_like, shape (N, 3)
        electric field per unit voltage on rf electrode, at N points.
        In units of ( V / (scale * m) ) / V = 1/(scale * m).
        (e.g. for scale = 1e-6, this is 1/um.)
    v_rf : number
        rf amplitude, in volts
    mass : number
        ion mass, in kg
    rf_omega : number
        rf angular frequency
    scale : number
        length unit used for e_field_arr, in meters

    Returns
    -------
    pseudopot_arr : ndarray, shape (N,)
        pseudopotential in volts at those N points.
    """
    e_field_ndarr = np.array(e_field_arr)
    e_field_squared_arr = np.sum(e_field_ndarr**2, axis=1)

    # phi_pseudo = q E^2 / (4 m omega_rf^2)
    return ELEM_CHARGE * (e_field_squared_arr * v_rf**2) \
        / (4 * mass * rf_omega**2) \
        * 1/scale**2


def import_unit_potentials(
    path,
    name,
    rf_electrodes,
    v_rf,
    mass,
    rf_omega,
    scale,
    verbose=True,
):
    """
    Import speciified grid files and return interpolators for each
    electrode giving the electric potential assuming 1V applied on the
    electrode and 0V on all others, except for the RF electrodes, for
    which we give the pseudopotential.

    Grid files must be named as STEM_1.txt, STEM_2.txt, ...

    Grid files must contain tab-separated values on each line,
    with the schema

        x, y, z, V, E_x, E_y, E_z

    where x, y, z are position coordinates (in meters * `scale`),
    V is the voltage (in volts), and E_{x,y,z} gives electric fields
    in volts / (meters * scale).

    Rows must be sorted first by z, then y, then finally x
    (all in ascending order).

    Parameters
    ----------
    path : string
        path to the grid files
    name : string
        name stem of grid files (STEM above)
    rf_electrodes : iterable
        list of rf electrodes (by grid file number)
    v_rf : number
        amplitude of rf, in volts
    mass : number
        ion mass, in kg
    rf_omega : number
        *angular* frequency of rf, in rad/s
    scale : number
        unit of length in grid files, in meters
    verbose : boolean
        whether to print progress of import

    Returns
    -------
    interpolators_dict : dict
    spaces : tuple
    grid_file_data_dict : dict
    """

    cwd = os.getcwd()
    os.chdir(path)
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    os.chdir(cwd)

    interpolators_dict = {}
    grid_file_data_dict = {}

    for fname in files:
        grid_file_name_regex = r'^{}_(\d+)\.txt$'.format(name)
        regex_match = re.match(grid_file_name_regex, fname)

        # not a grid file
        if regex_match is None:
            continue

        # flag for whether we've stored the lattice coordinates x, y, z
        spaces_known = False

        electrode_num = int(regex_match.group(1))
        if verbose:
            print('Importing electrode {}'.format(electrode_num))

        full_path = os.path.join(path, fname)
        with open(full_path, 'r') as f:
            csv_data = csv.reader(f, delimiter='\t')
            grid_file_data = np.array(list(csv_data), dtype=np.float32)
            potential_arr = np.zeros((0, 4))  # x, y, z, (pseudo)potential

            # TODO refactor this out and test
            if electrode_num not in rf_electrodes:
                potential_arr = grid_file_data[:, 0:4]
            else:
                e_field_arr = grid_file_data[:, 4:7]
                pseudopot = compute_pseudopot(
                    e_field_arr, v_rf, mass, rf_omega, scale)

                potential_arr = np.concatenate(
                    [grid_file_data[:, 0:3], pseudopot.reshape(-1, 1)],
                    axis=1
                )

            if not spaces_known:
                spaces, dimensions = get_spaces(grid_file_data)
                spaces_known = True

            interp_data = potential_arr[:, 3].reshape(dimensions, order='F')
            interpolator = RegularGridInterpolator(spaces, interp_data)
            interpolators_dict[electrode_num] = interpolator
            grid_file_data_dict[electrode_num] = grid_file_data

    if verbose:
        print('Potentials imported. (dev version)')

    return grid_file_data_dict, interpolators_dict, spaces, dimensions


def cartesian_product_ysort(xpts, ypts, zpts):
    """
    Given sorted 1D arrays xpts, ypts, zpts,
    return the Cartesian product xpts * ypts * zpts,
    sorted first by y, then z, then x.
    """
    return np.roll(
        np.array(
            np.meshgrid(ypts, zpts, xpts)
        ).reshape(3, -1).T,
        1,
        axis=1,
    )


def rotate_coeffs(theta, aa, ab, bb):
    """
    Given potential with coefficients aa, ab, bb, return the coefficients
    (aanew, abnew, bbnew) that produce the same potential, but tilted by
    theta (radians) CLOCKWISE in the ab-plane. That is, the original b-axis
    is now tilted theta toward the a-axis.
    """
    sin = np.sin
    cos = np.cos

    aanew = aa*cos(theta)**2 + bb*sin(theta)**2 + 0.5*ab*sin(2*theta)
    abnew = ab*cos(2*theta)  + (bb-aa)*sin(2*theta)
    bbnew = bb*cos(theta)**2 + aa*sin(theta)**2 - 0.5*ab*sin(2*theta)

    return [aanew, abnew, bbnew]


# Used as a default value for default constraints when generating fits
#   so that not all low-order coefficients must be specified every time
default_constraints = [
    ('x', 0), ('y', 0), ('z', 0),
    ("xx", 0), ("xy", 0), ("xz", 0), ("yy", 0), ("yz", 0), ("zz", 0)
]


class VoltageGeneration(object):

    # Sets up the basic requirements for the generation of voltages
    def __init__(
            self, grid_path, grid_name, electrode_grouping, rf_electrodes,
            order=4, f_cons=default_constraints, custom_fit=None, m=40*amu,
            rf_omega=2.*np.pi*46.E6, v_rf=[56], scale=1E-6,
            fit_ranges=[5e-6, 10e-6, 5e-6], rf_axis='y', verbose=True
    ):
        """
        Initialize a voltage generation object based on specified grid
        files, which must be named `<GRIDNAME>_<ELECTRODE_NUM>.txt`.

        All electrodes are specified by their grid file numbers.

        The grid files must follow the schema described in
        VoltageGeneration.import_unit_potentials (see above).

        Parameters
        ----------
        grid_path : string
            path to grid files
        grid_name : string
            name stem of grid files
        electrode_grouping : iterable<iterable<int>>
            list of lists giving electrodes tied together.
        rf_electrodes : iterable<int>
            iterable of rf electrodes (only one is supported at the moment)
        order : int
            maximum polynomial order to do fits with
        f_cons : ???
            default constraints to use in all fits unless overridden
        custom_fit : ???
            ???
        m : number
            ion mass, in kg
        rf_omega : number
            angular frequency of rf, in rad/s
        v_rf : iterable
            rf voltage amplitudes (currently must be 1-element list)
        scale : number
            length units of grid files, in meters (typically 1e-6)
        fit_ranges : array_like, shape (3,)
            (half the) dimensions of region in which to do polynomial fits
        rf_axis : one of {'x', 'y', 'z'}
            trap axis (currently only supports 'y'), TODO
        verbose : boolean
            whether to print information during grid file loading
        """

        if len(rf_electrodes) == 0:
            raise ValueError("Need at least one rf electrode!")
        elif len(rf_electrodes) > 1:
            raise NotImplementedError("Only one rf electrode supported!")
        self.rf_electrodes = tuple(rf_electrodes)

        self.m = m
        self.rf_omega = rf_omega
        self.v_rf = v_rf[0]
        self.scale = scale
        self.trap_length_scale = 5e-6 / scale

        self.grid_file_data, self.unit_potentials, self.spaces, self.dimensions = import_unit_potentials(
            grid_path, grid_name, self.rf_electrodes, self.v_rf, self.m,
            self.rf_omega, self.scale, verbose=verbose)
        self.xs, self.ys, self.zs = self.spaces

        self.fit_ranges = np.array(fit_ranges) / self.scale

        if rf_axis not in ['x', 'y', 'z']:
            raise ValueError('rf axis not x, y, or z')
        if rf_axis != 'y':
            raise NotImplementedError('support for non-y axes not yet built')
        self.rf_axis = rf_axis

        self.electrode_grouping = electrode_grouping
        self.default_constraints = f_cons
        if custom_fit is None:
            self.fit_functions = self.generate_fit_functions(order=order)
        else:
            self.fit_functions = self.createCustomFit(custom_fit, f_cons)
        self.fit_functions['const'] = lambda R: 1  # DC offset

        # TODO: rename or otherwise make clearer
        self.function_map = {k: ind for (ind, k) in enumerate(self.fit_functions.keys())}
        self.Fvec = [self.fit_functions[k] for k in self.fit_functions.keys()]
        self.evalFvec = lambda Rin: [Fs(Rin)*np.ones(np.shape(Rin)[1]) for Fs in self.Fvec]

        self.coeff_cache = {}
        self.prev_voltages = None

    def set_rf_voltage(self, v_rf):
        if self.grid_file_data is None:
            self.grid_file_data, self.unit_potentials, self.spaces, self.dimensions = import_unit_potentials(
            grid_path, grid_name, self.rf_electrodes, v_rf, self.m,
            self.rf_omega, self.scale, verbose=verbose)
        else:
            self.unit_potentials = {}

            for k, v in self.grid_file_data.items():
                potential_arr = np.zeros((0, 4))  # x, y, z, (pseudo)potential

                # TODO refactor this out and test
                if k not in self.rf_electrodes:
                    potential_arr = v[:, 0:4]
                else:
                    e_field_arr = v[:, 4:7]
                    pseudopot = compute_pseudopot(
                        e_field_arr, v_rf, self.m, self.rf_omega, self.scale)

                    potential_arr = np.concatenate(
                        [v[:, 0:3], pseudopot.reshape(-1, 1)],
                        axis=1
                    )

                interp_data = potential_arr[:, 3].reshape(self.dimensions, order='F')
                interpolator = RegularGridInterpolator(self.spaces, interp_data)
                self.unit_potentials[k] = interpolator
        self.v_rf = v_rf

    def reload_gridfiles(self, grid_path, grid_name, rf_electrodes=None, v_rf=None, mass=None, rf_omega=None, scale=None, verbose=True):
        if rf_electrodes:
        	self.rf_electrodes = rf_electrodes
        if v_rf:
        	self.v_rf = v_rf
        if mass:
        	self.m = mass
        if rf_omega:
        	self.rf_omega = rf_omega
        if scale:
        	self.scale = scale

        self.grid_file_data, self.unit_potentials, self.spaces, self.dimensions = import_unit_potentials(
            grid_path, grid_name, self.rf_electrodes, self.v_rf, self.m,
            self.rf_omega, self.scale, verbose=verbose)

    def is_coord_in_bounds(self, position):
        """
        Check whether the given coordinate is within the buonds
        of our grid files.

        Parameters
        ----------
        position : array_like, shape (3,)
            Coordinates to check.

        Returns
        -------
        in_bounds : boolean
            True if within bounds in all three dimensions, else false.
        """
        in_x, in_y, in_z = [
            min(self.spaces[i]) <= position[i] <= max(self.spaces[i])
            for i in range(3)
        ]

        return in_x and in_y and in_z

    def evaluate_f_vector(self, positions):
        """
        Evaluate the basis functions in self.Fvec (typically trivariate
        monomials) on some set of 3D coordinates.

        Parameters
        ----------
        positions : array_like, shape (k1, k2, ..., kn, 3)
            A stack of 3D coordinates (i.e. a reshaped matrix
            of 3D coordinates, length k1 * k2 * ... * kn)

        Returns
        -------
        f_tensor : ndarray, shape (k1, k2, ..., kn, len_fvec)
            A correspondingly shaped stack of vectors [f1(r), f2(r), ...]
            giving the value of each function in self.Fvec applied
            to each 3D coordinate.
        """
        positions_arr = np.array(positions)

        # positions as a list of 3D coordinates
        positions_matrix = positions_arr.reshape(-1, 3)

        f_matrix = np.array([
            [fit_func(position) for fit_func in self.Fvec]
            for position in positions_matrix
        ])

        num_fs = len(self.Fvec)
        f_tensor = f_matrix.reshape(positions_arr.shape[:-1] + (num_fs,))

        return f_tensor

    def set_electrode_grouping(self, electrode_grouping):
        """
        Allows the electrode grouping to be changed,
        so that we don't need to reload grid files
        if we want to change electrode grouping.
        """
        self.electrode_grouping = electrode_grouping
        self.prev_voltages = None

    def compute_potential(self, points, electrode_voltages):
        """
        Evaluate the electric potential at specified points given
        specified electrode voltages.

        Parameters
        ----------
        points : array_like, shape (3,) or (N, 3)
            The points at which to evaluate the potential function.
        electrode_voltages : iterable of pairs
            An iterable of pairs (electrode_num, voltage) specifying
            the voltage on each electrode. Omitted electroes are assumed
            to be at zero volts.

        Returns
        -------
        potential : ndarray
            1D array giving electric potential at the specified points.
        """
        points_arr = np.array(points).reshape(-1, 3)

        if len(points_arr.shape) > 2:
            raise ValueError("`points` not a shape ")
        potential = np.zeros((len(points_arr), ))
        for electrode_num, voltage in electrode_voltages:
            potential += voltage * self.unit_potentials[electrode_num](points_arr)
        return potential

    def compute_fitted_potential(self, points, R, coeffs):
        """
        Given the Taylor expansion coefficient fit values of the potential
        around some point R, find the ideal (fitted) potential at some points.

        Parameters
        ----------
        points : array_like, shape (N, 3)
            N points at which to compute the fitted potential,
            in grid file units (meter * self.scale)
        R : array_like, reshapable to (3,)
            expansion point
        coeffs : array_like
            fit coefficient values, one per basis function, in the order
            given by self.Fvec

        Returns
        -------
        fitted_potentials : ndarray, shape (N,)
            Value of the fitted potential function at the N points.
        """
        f_matrix = self.evaluate_f_vector(points - R)
        return f_matrix.dot(coeffs)

    # TODO clean up this function
    # Given a point to perform the expansion around, compute the minimum
    #   of the pseudopotential
    def findPseudoMinima(self, R, tol=EPSILON):
        coeffs = self.get_electrode_coefficients(self.rf_electrodes, R)

        def potential_function(pt):
            return self.compute_fitted_potential(pt, R, coeffs)
            # return np.dot(np.array(self.evalFvec((pt-R).T)).T, coeffs)

        result = sciopt.minimize(potential_function, R, tol=tol)
        return result.x

    # TODO clean up this function
    def findPseudoMinimaRadial(self, R, tol=EPSILON):
        """
        Given point R, find the location of the minimum of the polynomial
        approximation of the pseudopotential when restricted to the radial
        plane.
        """
        R = np.array(R)
        r = R[0]
        coeffs = self.get_electrode_coefficients(self.rf_electrodes, R)

        def potential_function(point):
            return self.compute_fitted_potential(point, R, coeffs)
            # return np.dot(np.array(self.evalFvec((point - R).T)).T, coeffs)

        def potential_function_radial(point):
            return potential_function(np.array([point[0], r[1], point[1]]))

        result = sciopt.minimize(potential_function_radial, [r[0], r[2]], tol=tol).x
        return [result[0], r[1], result[1]]

    # TODO clean up this function
    # Compute the pseudopotential minima sampled on a discrete grid of points
    def findPseudoMinimaDiscrete(self, R):
        xpts = self.xs - np.mean(self.xs)
        ypts = self.ys - np.mean(self.ys)
        zpts = self.zs - np.mean(self.zs)

        allPts = cartesian_product_ysort(xpts, ypts, zpts) + R
        elconfig = []
        for el in self.rf_electrodes:
            elconfig.append([el, 1])
        pots = self.compute_potential(allPts, elconfig)
        minloc = np.argmin(pots)
        return allPts[minloc]


    # TODO find a way to make this function faster
    # TODO allow for restricting the search grid to more reasonable extents
    def find_total_minimum_discrete(self, voltages):
        xpts = self.xs - np.mean(self.xs)
        ypts = self.ys - np.mean(self.ys)
        zpts = self.zs - np.mean(self.zs)

        trap_center = [0., 0., 50.]

        allPts = cartesian_product_ysort(xpts, ypts, zpts) + trap_center
        elconfig = []
        for el in self.rf_electrodes:
            elconfig.append([el, 1]) #For the RF, the real pseudopotential amplitude is already taken into account

        for electrode_num, voltage in enumerate(voltages):
            elconfig.append([electrode_num+1, voltage])

        pots = self.compute_potential(allPts, elconfig)
        print(pots)


        minloc = np.argmin(pots)

        return allPts[minloc]


    # TODO clean up this function more
    # rename to `get_electrode_group_coefficients`?
    # or make it return a matrix, shape (n_electrodes, len_fvec)
    def get_electrode_coefficients(self, electrodes, R):
        """
        Given a list of electrodes, find the Taylor coefficients of the
        resulting potential in a neighborhood of a specified point R
        by fitting to values in a lattice around R spaced as the grid files
        and extending amounts specified by self.fit_ranges.

        Parameters
        ----------
        electrodes : iterable<int>
            Electrodes tied together, specified as grid file numbers.
        R : array_like, shape (3,)
            Taylor expansion point.

        Returns
        -------
        coeff_fit : ndarray, shape (len_fvec,)
            Fitted coefficients for the Taylor expansion of the electric
            potential around R when specified electrodes are have unit
            voltage applied, specified in the order in self.Fvec

        Raises
        ------
        ValueError
            If the given coordinate is not in bounds.
        """
        r = np.array(R).reshape(-1, 3)
        assert r.shape == (1, 3)

        if not self.is_coord_in_bounds(r.reshape(3)):
            raise ValueError("Coordinates {} not in bounds".format(R))

        # TODO make this a proper LRU cache
        # check the cache
        myKey = str(electrodes)+str(r)
        if myKey in self.coeff_cache.keys():
            return self.coeff_cache[myKey]

        coords_array = []
        for space, r_coord, fit_range in zip(self.spaces, r[0], self.fit_ranges):
            in_range_mask = abs(space - r_coord) <= fit_range
            coords_array.append(space[in_range_mask])
        xpts, ypts, zpts = coords_array

        # cartesian product, but sorted first by y, then z, then x.
        # shape = (-1, 3)
        lattice_points = cartesian_product_ysort(xpts, ypts, zpts)

        el_vec = [[k, 1] for k in electrodes]
        pots = self.compute_potential(lattice_points, el_vec)  # This is a 1D list

        # To get coefficients a[i] for basis functions f[i] : R3 => R
        # producing a fit to a function phi : R3 => R, as
        #
        # phi ~ sum_i a[i] * f[i](r)
        #
        # given values on some set of points r[j],
        # define matrix F[i, j] = f[i](r[j])
        # so that F.T[j, i] . a[i] ~ phi(r[j]).
        #
        # Intuitively, we should be able to find a[i] by some matrix inverse.
        # Indeed, we use the matrix pseudoinverse A^+, which by definition
        # is the matrix such that for any least squares problem Ax ~ b
        # x = A^+ b is the solution.

        f_matrix_transpose = self.evaluate_f_vector(lattice_points - r)
        f_pinv = la.pinv(f_matrix_transpose)  # pinv commutes with transposition
        coeff_fit = np.dot(f_pinv, pots)

        self.coeff_cache[myKey] = coeff_fit
        return coeff_fit

    # TODO clean up this function

    # Given a point R, generates a matrix C which maps voltages
    #   to observed coefficients at the point R, where the coefficients
    #   of interest are the ones in cons.
    # Returned matrix has dimensions of N_coeff x N_electrodes
    def generateCMatrixPt(self, R, cons):
        Cpt = np.zeros((len(self.electrode_grouping), len(cons)))
        for i, el in enumerate(self.electrode_grouping):
            all_coeffs = self.get_electrode_coefficients(el, R)
            Cpt[i] = [all_coeffs[self.function_map[k]] for k, _ in cons]
        return Cpt.T


    # TODO clean up this function

    # Given a vector of points R, and constraints at those points, returns a matrix
    #   C which maps voltages to observed coefficients at those points.
    # If n_pts > 1, cons is an array of arrays of tuples, each tuple is one constraint
    # Returned matrix has dimensions of N_constraints x N_electrodes
    def generateCMatrix(self, R, cons):
        C = np.zeros(len(self.electrode_grouping))
        for r, c in zip(R, cons):
            # Should this be vstack? TODO: Fix this?
            C = np.vstack((C, self.generateCMatrixPt(r, c)))

        return C[1::]

    # TODO clean up this function
    # Change dV constraint to be an absolute voltage rather than a distance
    # Add/refine a constraint which mimizes the total size of the voltage vector
    # Better documentation/reasoning for the weighting of higher-order coefficients
    def findControlVoltages(
            self, R, cons=default_constraints, bnds=(-np.inf, np.inf),
            tol=1E-15, pseudoCorrection=False, dv_weight=0, max_dv=1E6,
            independent=True, fixed_voltages = [], normvec=None, epss=1e-4
    ):
        """
        Given an expansion points and a set of target coefficients, solves the least-
          squares problem for Cv=y for v, subject to additional constraints to keep
          voltages within acceptable ranges and keep waveforms smooth during transport.

        Parameters
        ----------
        R : array_like, reshapable to (3,)
            expansion point
        cons : ???
            set of constraints on the coefficients of the Taylor series
        bnds : tuple
            maximum range of voltages
        tol : ???
            ???
        pseudoCorrection : boolean
            not implemented; would be a flag to include the pseudopotential in the fit
        dv_weight : float
            weighting factor applied to the Euclidian distance between voltage sets
        max_dv : float
            ???
        independent : boolean
            when true, do not use the previous voltage solution in the constraints
        independent : list of tuples
            each tuple contains the (index, voltage) for a constrained electrode

        Returns
        -------
        ret_val.x : ndarray
            Voltage arrray found using least squares optimization.
        """

        # Prepare the specified fixed voltaes for insertion into the voltage array
        #  during optimization
        fixed_voltages.sort(key=lambda cond: cond[0])

        fixed_voltage_indices = [cond[0] for cond in fixed_voltages]
        fixed_voltage_values = [cond[1] for cond in fixed_voltages]

        for i in range(len(fixed_voltage_indices)):
            fixed_voltage_indices[i] -= i

        # JUPDATE: Is there a safer way to handle cases like when you switch which electrodes are constrained between fits?
        if (independent
                or self.prev_voltages is None
                or len(self.prev_voltages) != len(self.electrode_grouping, ) - len(fixed_voltages)):
            self.prev_voltages = np.zeros((len(self.electrode_grouping, ) - len(fixed_voltages)))

        # Ensures that BOTH the default coefficients and the supplied coefficients
        #   are used in the generating of the C matrix; also sorts the coefficients
        #   so they're in the correct, ascending order
        tcons = []
        for conset in cons:
            # NOTE: Some older notebooks could throw an error here; make sure that
            #  self.default_constraints is a list of tuples and not a list of lists
            tcons += [sorted(dict(self.default_constraints + conset).items(),
                key=lambda x: int(self.encode_string(x[0])))]
        C = self.generateCMatrix(R, cons=tcons)

        # not implemented; intended for doing fits including the pseudopotential
        # pseudoFit = np.zeros(1)

        # Converts the target coefficient constraints in cons, which can include
        #   multiple points, into the vector y
        target_coeffs = np.zeros(1)
        order_coeffs = np.zeros(1)

        for i in range(np.shape(R)[0]):
            target_cons = {}
            # Populate the vector with the constraints
            for (k, v) in tcons[i]:
                target_cons[k] = v

            # Masks the constraint vector to only hold the desired coefficients,
            #   instead of all of the coefficients
            mask_vec = np.zeros((len(self.fit_functions.keys()), 1), dtype=bool)
            target_vec = np.zeros((len(self.fit_functions.keys()), 1))
            order_vec = np.zeros((len(self.fit_functions.keys()), 1), dtype=int)

            for k in target_cons.keys():
                # canonical integer associated with the constraint
                # (e.g. 'xx' might be the 4th constraint)
                constraint_index = self.function_map[k]
                target_vec[constraint_index] = target_cons[k]
                mask_vec[constraint_index] = True
                order_vec[constraint_index] = 0 if k == 'const' else len(k)

            target_vec = np.reshape(target_vec[mask_vec], (-1, 1))
            order_vec = np.reshape(order_vec[mask_vec], (-1, 1))
            target_coeffs = np.vstack((target_coeffs, target_vec))
            order_coeffs = np.vstack((order_coeffs, order_vec))

        target_coeffs = target_coeffs[1::]

        # we normalize coefficients based on the order of their monomial
        # quadratic gets multiplied by d^2, quartic by d^4, etc.
        # where d = self.trap_length_scale is a characteristic length scale
        # of the trap
        # intuition: the badness of an error should correspond to errors in
        # the generated potential; also, this normalization makes our code
        # dimensionally correct.
        normalization_vec = np.power(self.trap_length_scale, order_coeffs[1::])

        #print(order_coeffs[1::])
        #print(normalization_vec[-1]*10)

        if normvec is not None:
            normalization_vec *= normvec
            
        def cost(voltages):
            dv = la.norm(self.prev_voltages.reshape(-1)-voltages.reshape(-1))
            total_cost = dv*dv_weight
            
            voltages = np.insert(voltages, fixed_voltage_indices, fixed_voltage_values)
            voltages = voltages.reshape(len(voltages), -1)
            realized = C.dot(voltages)
            sq_diff = la.norm(
                (realized.reshape(-1)-target_coeffs.reshape(-1))*normalization_vec.reshape(-1)
            )
            total_cost = total_cost + sq_diff

            # voltage_norm = la.norm(voltages)
            # total_cost += voltage_norm*1e-6

            return total_cost

        # JUPDATE: Is this maximum delta voltage even relevant any more?
        delta_v = lambda x: max_dv - np.max(np.abs(x-self.prev_voltages))
        cdict = ({'type': 'ineq', 'fun': delta_v})

        # LUPDATE: option to set different bounds for different electrode groupings
        boundss = np.repeat([bnds], len(self.electrode_grouping)-len(fixed_voltages), axis=0)
        ret_val = sciopt.minimize(
            cost,
            method='SLSQP',
            x0=self.prev_voltages,
            #x0=np.zeros((len(self.electrode_grouping, ))),
            tol=tol,
            constraints=cdict,
            options={'maxiter':2000, 'eps':epss},
            bounds=boundss,
        )

        if not ret_val.success:
            print("Optimization failed after %i iterations."%(ret_val.nit))

        self.prev_voltages = ret_val.x

        # Insert fixed voltages into array just before returning
        output_voltages = np.insert(ret_val.x, fixed_voltage_indices, fixed_voltage_values)
        realized_coeffs = C.dot(output_voltages)
        print('Target, Realized Coeffs:\n', np.concatenate([np.array(list(itertools.chain.from_iterable(tcons))),
                                                            realized_coeffs.reshape(-1,1)], axis=1))
        print('Final cost value: ', ret_val.fun)
        print('Number of iterations: ', ret_val.nit, '\n')
        return output_voltages

    # TODO clean up this function
    # Given a list of electrodes tied together and voltages to apply,
    #   this function fits the potentials in a cube of side length given
    #   sampled over pts_side^3 points uniformly through the cube to fit functions.
    def getCoefficientsForElVoltage(self, configuration, R):
        c = configuration[0]
        coeffs = c[1]*self.get_electrode_coefficients(c[0], R)
        for c, v in configuration[1::]:
            coeffs += v*self.get_electrode_coefficients(c, R)
        return coeffs

    def get_config_coefficients(self, config, R):
        """
        """
        coefficients = 0
        for electrode, voltage in config:
            coefficients += voltage * self.get_electrode_coefficients(c, R)
        return coefficients

    # TODO clean up this function
    # Given a point R and a list of electrodes of interest, computes the
    #   coefficients specified in coeffs (in plain text e.g. 'xz') and prints
    #   them. Useful for debugging purposes.
    def printCoefficients(self, R, electrodes, coeffs, printing=True):
        """
        Given a point, a list of electrodes, and a set of monomials,
        return the coefficients on those monomials in the Taylor expansion
        of the electric potential around the point.

        Parameters
        ----------
        R : array_like, shape (3,)
            point around which to do expansion
        electrodes : iterable
            electrodes for which to find expansion coefficients
        coeffs : iterable<str>
            coefficients of interest
        printing : boolean, optional
            whether to print the results (useful for debugging)

        Returns
        -------
        ?????
        """
        R = np.array(R)

        retlist = []
        for electrode in electrodes:
            coe = np.zeros((0, 1))
            if printing:
                print("Electrode", electrode)
            C = self.get_electrode_coefficients([electrode], R)
            for co in coeffs:
                if printing:
                    print(co, C[self.function_map[co]])
                coe = np.vstack((coe, C[self.function_map[co]]))
            retlist.append(coe)
        return retlist

    # TODO clean up this function
    # Given a point R and voltages to apply, computes the observed coefficients
    #   specified in coeffs and returns them. Useful for simulations.
    def voltageToCoefficients(self, R, voltages, coeffs):
        R = np.array(R)
        config = self.ungroup_configuration(voltages)
        retcoeffs = np.zeros((len(coeffs), ))
        for el, v in config:
            coe = np.zeros((0, ))

            # TODO terrible name, fix
            C = self.get_electrode_coefficients([el], R.reshape(3, -1).T)*v
            for co in coeffs:
                coe = np.append(coe, C[self.function_map[co]])
            retcoeffs += coe.reshape(-1)
        return retcoeffs

    def compute_potential_fit(self, R, points, electrode_config):
        """
        Given a Taylor expansion point R and some electrode configuration,
        compute the (Taylor-fitted-at-R) potential at a series of points.

        Parameters
        ----------
        R : array_like, shape (3,)
            Taylor expansion point
        points : array_like, shape (K, 3)
            K points at which to find the (fitted) potential.
        electrode_config : iterable
            Electrode configuration (assignments of voltages to electrodes)
            given as an iterable of pairs (electrode_num, voltage)

        Returns
        -------
        potentials : ndarray, shape (K,)
            Electric potential at each specified point computed
            from the fit at R.
        """
        R = np.array(R).reshape(1, 3)

        potentials = np.zeros((len(points),))
        for electrode, voltage in electrode_config:
            # basis function coefficients in fit at R
            electrode_coeffs = self.get_electrode_coefficients([electrode], R)

            # values of the basis functions
            electrode_f_matrix = self.evaluate_f_vector(points - R)
            potentials += voltage * electrode_f_matrix.dot(electrode_coeffs)

        return potentials

    # Given a point R to expand around and an electrode configuration
    #   el_pots, computes the potential at all points according to the
    #   fit to the potential data
    def computePotentialFit(self, R, points, el_pots):
        raise DeprecationWarning('Use reimplementation `compute_potential_fit`'
                                 'instead')

        R = np.array(R)
        pot_out = np.zeros((len(points), ))
        for el, Vapp in el_pots:
            c = self.get_electrode_coefficients([el], R.reshape(3, -1).T)
            Fmtrx = np.array(self.evalFvec((points-R).T))
            print(Fmtrx.T)

            pot_out += Fmtrx.T.dot(c)*Vapp
            print(el, Fmtrx.T.dot(c)*Vapp)

        return pot_out

    def compute_total_potential_axes(self, R, config, printing=True):
        """
        Given point R and a DC electrode configuration, compute the
        total potential (i.e. DC and pseudopotential) and use this to
        determine the potential minimum, principal axes, and trap
        frequencies for each axis.

        Parameters
        ----------
        R : ?
            Point around which to perform polynomial expansion.
        config : ?
            Grouped electrode configuration
        """
        # config = np.array([list(self.electrode_grouping), config]).T
        # coeffs = self.getCoefficientsForElVoltage(config, R)
        # if printing:
        #     print("coeffs without the RF:", coeffs)

        config = np.vstack((config, [self.rf_electrodes, 1]))
        coeffs = self.getCoefficientsForElVoltage(config, R)
        if printing:
            print("coeffs with the RF:", coeffs)

        xx = coeffs[self.function_map['xx']]
        xy = coeffs[self.function_map['xy']]
        xz = coeffs[self.function_map['xz']]
        yy = coeffs[self.function_map['yy']]
        yz = coeffs[self.function_map['yz']]
        zz = coeffs[self.function_map['zz']]

        hessian = np.array([
            [xx, .5*xy, .5*xz],
            [.5*xy, yy, .5*yz],
            [.5*xz, .5*yz, zz]
        ])

        eigvals, v = scipy.linalg.eigh(hessian)
        eigvecs = v.T
        sec_freqs = np.sqrt(2*eigvals*ELEM_CHARGE/(self.m*self.scale**2))/(2*np.pi)

        if printing:
            # print("Hessian:")
            # print(hessian)

            print("Frequencies in Hz:", sec_freqs)
            # print("Corresponding axes (as row vectors):\n", eigvecs)
            print("Tilt of each eigenvector relative to +x in x-z plane:\n", \
              (np.arctan(eigvecs[:, 2]/eigvecs[:, 0])*180/np.pi))
            print("Each eigenvector in spherical coordinates (theta, phi):\n", \
              (np.arccos(eigvecs[:, 2])*180/np.pi),\
                 np.arctan(eigvecs[:, 1]/eigvecs[:, 0])*180/np.pi)

        # return hessian, sec_freqs, eigvecs
        return coeffs

    def tilt_with_pseudo(self, R, theta_target, uu, delta=3):
        """
        Find dc potential constraints that produce a combined dc+rf
        pseudopotential with desired radial tilt angle and axial frequency.

        In particular, if we label our axes u, b, a (with u the trap axis)
        such that 'uba' is either has the same parity as 'xyz',
        we will tilt the the b principal axis toward a.

        Parameters
        ----------
        R : array_like, shape (3,)
            expansion point of interest
        theta_target : number
            desired tilt angle, of b axis toward a, in radians
        uu : number
            axial quadratic constraint
        delta : number
            used to satisfy Laplace's equation; untilted constraint bb
            will be set to delta * uu and constraint aa to -(1+delta)*uu

        Returns
        -------
        (aa, ab, bb): numbers
            constraints for the tilted potential
        """

        # names of the radial coefficients
        # these correspond to 'aa', 'ab', 'bb' later in this function
        if self.rf_axis == 'y':
            coeff_name_strings = 'xx', 'xz', 'zz'
        elif self.rf_axis == 'z':
            coeff_name_strings = 'yy', 'xy', 'xx'
        else:
            coeff_name_strings = 'zz', 'yz', 'yy'

        # coefficients for untilted potential
        # delta ensures we satisfy Laplace's equation
        bb = delta * uu
        aa = - (1 + delta) * uu
        ab = 0

        rf_radial_coefficients = self.printCoefficients(
            R, self.rf_electrodes, coeff_name_strings, printing=False)[0]
        rfaa, rfab, rfbb = rf_radial_coefficients[:, 0]

        def tilt_error(theta_tilt):
            # coefficients for tilted potential; think 'aa-tilted', etc.
            aat, abt, bbt = rotate_coeffs(theta_tilt, aa, ab, bb)

            # find tan(2theta), where theta is the actual tilt angle
            # produced by [aat, abt, bbt]
            tan2theta = (abt + rfab)/((rfbb + bbt) - (rfaa + aat))
            err = (np.tan(2 * theta_target) - tan2theta)**2
            return err

        theta_opt = sciopt.minimize_scalar(tilt_error).x
        return rotate_coeffs(theta_opt, aa, ab, bb)

    def ungroup_configuration(self, voltages_grouped):
        """
        Given a grouped voltage configuration, returns a list of individual
        electrode voltages, ungrouped.
        """
        ungrouped = []
        for i, v in enumerate(voltages_grouped):
            for el in self.electrode_grouping[i]:
                ungrouped.append([el, v])
        return ungrouped

    def createCustomFit(self, terms, default_constraints):
        """
        Given a list of tuples, coeffs, generates a fit function dict
          containing all of those coefficients.
        This is used to specify a fit function which only utilizes
          certain monomials, rather than all monomials up to Nth order
        Note that you can specify in terms the only terms which are not
          constrained by default, to cut down on repetitive declaration of terms
        """
        x = lambda R: R[0]
        y = lambda R: R[1]
        z = lambda R: R[2]
        fdict = {'x': x, 'y': y, 'z': z}

        dictout = collections.OrderedDict()
        for ident, _ in default_constraints:
            flist = []
            for c in ident:
                flist += [fdict[c]]
            dictout[ident] = functools.reduce(lambda a, b: lambda r: a(r)*b(r), flist)
        for ident in terms:
            flist = []
            for c in ident:
                flist += [fdict[c]]
            dictout[ident] = functools.reduce(lambda a, b: lambda r: a(r)*b(r), flist)
        return dictout

    def generate_fit_functions(self, order=4):
        """
        Generates all x^s*y^t*z^u s.t. s+t+u = i, but elements which are
          identical under a permutation are omitted, i.e. x*x*y is present,
          but x*y*x is not. Takes a maximum order (maximum i) as argument. 4 is quartic.
        Returns a dictionary that can be accessed via dictout["..."] where "..."
          is a string identifiying the monomial
        Note that the identifiers are all in nondecreasing order with x<y<z, so xyz
          would be present, but xzy and zxy would not
        """
        # Shorthand for making basis functions legible
        x = lambda R: R[0]
        y = lambda R: R[1]
        z = lambda R: R[2]

        dictout = collections.OrderedDict()
        for i in range(1, order+1):
            # Generates the combinations of a certain order
            flist = list(itertools.combinations_with_replacement([x, y, z], i))
            for f in flist:
                # Construct the identifying string
                identifier = ""
                for g in f:
                    if g == x:
                        identifier += "x"
                    if g == y:
                        identifier += "y"
                    if g == z:
                        identifier += "z"
                # Sets that value to be the product of the individual components
                dictout[identifier] = functools.reduce(lambda a, b: lambda r: a(r) * b(r), f)
        return dictout

    # Used to sort fitting functions
    def encode_string(self, s):
        return s.replace('x', '1').replace('y', '2').replace('z', '3').replace('const', '0')
