# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:14:31 2020

This runs a tracer program based off input text file

@author: a29white
"""

from pathlib import Path as pl
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator, griddata
from scipy.linalg import block_diag
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from matplotlib.path import Path
from datetime import datetime

##########################################################################
# Command Line parsing
##########################################################################

parser = argparse.ArgumentParser()

# File Parse
parser.add_argument('-f','--file', help='Input the file path')

# inlet and outlet files
parser.add_argument('-io','--inlet_outlet', 
                    help='Input the inlet/outlet file path')

# Time step
parser.add_argument('-t', '--timestep',type=float, 
        help='Input the time step size in seconds')

# Number of particles
parser.add_argument('-n', '--num', type=int, 
                    help='Input the number of particles')

# Output option
parser.add_argument('-o', '--output', help='Choose an output Option',
        choices=['csv', 'plot', 'count'])

# Sim Option
parser.add_argument('-i', '--interpolate', 
                    help='Interpolation option: linear, cubic or nearest',
                    choices=['linear', 'cubic', 'nearest'])

##########################################################################
# I/O Functions
##########################################################################
        
def getfilename(whichfile):
    """ 
    Get user input for txt file name
    """
    
    fname = input("Please input "+ whichfile +" file name ")
    return fname


def getpath():
    """
    Get i/o path from user and convert path object
    """
    
    if input("Input/Output to {}? y/n ".format(os.getcwd())) == 'n':
        path1 = input('Please input the full path of the i/o directory ')
        iodir = pl(path1)
        return iodir
    else:
        iodir = pl(os.getcwd())
        return iodir

    
def make_file(path, filename=None):
    """
    Make file path object to a file

    Arguements
    ----------
    path : str or pathlib object
        A string of the files directory or a path object of the directory
    filename: str
        name of the file to open
        
    return
    ------
    a pathlib object to the file
    
    """
    
    if filename == None:
        if isinstance(path, pl):
            return path
        else:
            return pl(path)
    else:
        if isinstance(path, pl):
            return path / filename
        else:
            return pl(path) / filename


def get_comsol_data(txt_file):
    """
    Get the data from the comsol output txt file
    
    Arguements
    ----------
    txt_file : pathlib or string
        txt file location of comsol output, pathlib object highly 
        suggested. make_file can build a pathlib object
    
    return
    ------
    data : list
        ardata : an ndarray of the x,y, x velocity, y velocity, speed
        names : the names of the columns as stated above
    """
    
    with open(txt_file, 'r') as f:
        data = f.readlines()
    
    columns = []
    header = []
    for line in data:
        if '%' in line:
            line = line[2:].rstrip()
            header.append(line.split('  '))
        else:
            columns.append(line.split())
            
    names = [i for i in header[8] if i !='']
    pos = [i + ' ' + '(' + header[7][-1] + ')' for i in names[:2]]
    names = names[2:]
    for i in range(len(pos)):
        names.insert(i,pos[i])
                
    ardata = np.array(columns)
    if type(ardata[0,0]) != float:
        ardata = ardata.astype(float)
        
    return [ardata, names]


def get_inlet_outlet(csv_file):
    """
    Get the data from the csv file containing inlet/outlet points
    
    Arguements
    ----------
    csv_file: pathlib or string
        csv file location of the inlet/outlet points, pathlib object 
        highly suggested. make_file can build a pathlib object
    
    return
    ------
    a list of inlet and outlet points
    """
    
    with open(csv_file, 'r') as f:
        data = f.readlines()
    
    io_arr = []
    for line in data:
        if 'x' not in line:
            io_arr.append(line.split())
        else:
            pass
            
    # inlets
    io_arr = np.array(io_arr)
    i_arr = io_arr[np.where(io_arr[:,2]==1),[0,1,3]]
    unique_inlet = np.unique(io_arr[:,2])
    inlet_list = [i_arr[np.where(i_arr[:,2] == i),:2].tolist() for
                         i in unique_inlet]
    # Outlets
    o_arr = io_arr[np.where(io_arr[:,2]==0),[0,1,3]]
    unique_outlet = np.unique(io_arr[:,2])
    outlet_list = [o_arr[np.where(o_arr[:,2] == i),:2].tolist() for
                         i in unique_outlet]
            
    return [inlet_list, outlet_list]


def output_csv(res_array, iodir):
    """
    Output the result to a new directory in the iodir
    
    Arguments
    ---------
    res_array : ndarray
        (n,z,6) n particles, z timesteps and 6 columns of values
        of interest
    iodir : pathlib object
        pathlib to the desired output directory
        
    outputs iodir/tracer_results_%time%
    file names are particle_%num%.csv
    """    
    
    run_time = datetime.now()
    dir_name = 'tracer_results_' + run_time.strftime('%d-%m-%y_%H-%M')
    os.mkdir(iodir / dir_name)
    
    for part in range(res_array.shape[0]):
        fname = 'particle_' + str(part) + '.csv'
        np.savetxt(iodir / dir_name / fname,
                   res_array[part,:,:], delimiter=',')
     
        
def in_out_tocsv(inlets, outlets, file):
    """
    Create a csv file from the user defined inlet/outlet
    
    Arguments
    ---------
    inlets : list
        list of ndarrays of the inlet indices
    inlets : list
        list of ndarrays of the outlet indices
    file : pathlib object
        Path to the file to save
    """
    
    def looper(tup, mat_build):
        arr_list = []
        for i in range(len(tup)):
            arr = mat_build((tup[i].shape[0],3))
            arr[:,0] = tup[i]
            arr[:,2] = i
            arr_list.append(arr)
        
        return np.vstack(arr_list)
    
    inarr = looper(inlets, np.ones)
    outarr = looper(outlets, np.zeros)
    tofile = np.vstack([inarr, outarr])
    np.savetxt(file, tofile, delimiter=',')


def in_out_read(file):
    """
    Create a csv file from the user defined inlet/outlet
    
    Arguments
    ---------
    file : pathlib object
        Path to the file to where the inlet/outlet data is saved
    
    returns
    -------
    list
        list of lists of ndarray's of the inlet/outlet indices
    """
    
    arr = np.genfromtxt(file, delimiter=',')
    ins = arr[np.where(arr[:,1] == 1),:]
    outs = arr[np.where(arr[:,1] == 0),:]
    
    numins = np.unique(ins[0,:,2])
    numouts = np.unique(outs[0,:,2])
    
    if numins.shape[0] == 1:
        inslist = ins[0,:,0].astype(int)
    else:
        inslist = []
        for i in numins:
            inslist.append(ins[0,np.where(ins[0,:,2] == i),0]
            .astype(int).ravel())
    
    if numouts.shape[0] == 1:
        outslist = outs[0,:,0].astype(int)
    else:
        outslist = []
        for i in numouts:
            outslist.append(outs[0,np.where(outs[0,:,2] == i),0]
            .astype(int).ravel())
    
    return [inslist, outslist]


##########################################################################
# Plotting
##########################################################################

def geoplot(data, figsize, border=None, in_out=None,
        triangles=None, t_indices = [], block=False):
    """
    Visualization of the geometry of the separator  
    
    Arguements
    ----------
    data: ndarray
        The comsol output file data for x and y
    figsize : tuple of ints 
        Specifying the width and height of the figure
    border : set
        x,y pairs of the border points
    in_out : list
        list of x,y pairs of the inlet/outlet points
    triangles : Delaunay triangulation object
        array of triangle vertices
    
    Displays a plot
    """
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(data[:,0],data[:,1], '.' , color='r',
            markersize=(30*figsize[0]*figsize[1])/data.shape[0])
    ax.set_title('Meshpoint geometry')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if border != None:
        for i, j in border:
            ax.plot(data[[i, j], 0], data[[i, j], 1], 'b')
            
    if in_out != None:
        circle_list = []
        for i in in_out:
            circle_list.append(plt.Circle(i, 0.01, color='g', fill=False))
            
        for circle in circle_list:
            ax.add_artist(circle)
            
    if triangles != None:
        ax.triplot(data[:,0], data[:,1], 
                   triangles.simplices[t_indices,:])
    
    fig.show(block)
            

def quivplot(data, figsize, block=False):
    """
    Vector field visualization
    
    Arguements
    ----------
    data: ndarray
        The comsol output file data for x, y, x vel, y vel
    figsize : tuple of ints 
        Specifying the width and height of the figure
    
    Displays a plot
    """
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    ax.quiver(data[:,0], data[:,1], data[:,2], data[:,3])
    ax.set_title('Vector Field (m/s)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    fig.show(block)


def plot_trajectories(p_trajectory, figsize, data, 
                      border=None, block=False):
    """
    Plot the particle paths
    
    Arguements
    ----------
    p_trajectory : ndarray
        (n,z,(x,y)) : n particles x,y coordinates at z points in time
    figsize : tuple of ints 
        Specifying the width and height of the figure
    Border : set
        x,y pairs of the border points
        
        
    Displays a plot
    """
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    
    for part in range(p_trajectory.shape[0]):
        ax.plot(p_trajectory[part,:,1], p_trajectory[part,:,2], 
                marker='*',ms= 1, color='b')
    
    ax.set_title('Particle Trajectory')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if border != None:
        for i, j in border:
            ax.plot(data[[i, j], 0], data[[i, j], 1], 'k')

    fig.show(block)


##########################################################################
# Geometric functions
##########################################################################

def inlet_outlet(boundary, inlet, outlet):
    """
    Define the inlets and outlets and ensure they are on the boundary.
    Implicit that there is at least one inlet and outlet
    
    Arguements
    ----------
    Boundary : list
        list of the border indices from the xy point data
    inlet : tuple
        tuple of ndarrays of data indices of inlet location
    outlet : tuple
        tuple of ndarrays of data indices of outlet location
        
    return
    ------
    a list of inlet and outlet array indices
    """
    
    # check for multiple inlets/outlets
    if len(inlet) == 1:
        inlet_set = {x for x in inlet[0]}
    else:
        inlet_set = {x for sub in inlet for x in sub}
    
    if len(outlet) == 1:
        outlet_set = {x for x in outlet[0]}
    else:
        outlet_set = {x for sub in outlet for x in sub}
    
    # Boundary set
    bound_set = {x for x in boundary}
    
    # check if in boundary
    if inlet_set.issubset(bound_set) and \
    outlet_set.issubset(bound_set):
        return [inlet, outlet]
    else:
        fps = [inlet_set.difference(bound_set),
               outlet_set.difference(bound_set)]
        print(fps)
        raise ValueError('are not on the boundary')


def alpha_shape(points, alpha=0.18, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    
    Arguements
    ----------
    points : ndarray
        (n,2) points.
    alpha : int
        alpha weight ie normalized radius surrounding points
    only_outer : boolean 
        value to specify if we keep only the outer border or 
        also inner edges.
        
    return
    ------
    edges : set
        (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
    tri : Delaunay triangulation object
        The delaunay triangulation of the separator
    """
    
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges
            if only_outer:
                # if both neighboring triangles are in shape, 
                #it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/
        # derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return [edges, tri]


def neighborhood_test(data, xlow, xhigh, ylow, yhigh):
    """
    Get the points in the data within the limits
    
    Arguements
    ----------
    data : ndarray
        (n,2) the x,y geometry of comsol output
    xlow : float
        the x lower limit
    xhigh : float
        the x higher limit
    xlow : float
        the y lower limit
    xhigh : float
        the y higher limit
        
    return
    ------
    neighborhood : ndarray
        (j,) an array of indexes in data for which the points lie
        within the x and y limits
    """
    neighborhood = np.where(np.logical_and(
        np.logical_and(data[:,0] <= xhigh, data[:,0] >= xlow),
        np.logical_and(data[:,1] <= yhigh, data[:,1] >= ylow)))
    
    return neighborhood


##########################################################################
# Main Simulation function
##########################################################################

def generate_particles(inlet_set, data, number_particles):
    """
    Generate random particles and place them in the inlet

    Arguements
    ----------

    inlet_list : tuple
        List of xy coordinates
    data : ndarray
        (n,2) x,y coordinates of the separator geometry
    number_particles : int
        Number of particles to simulate, note that there is 
        no check to ensure particles can fit in the interval

    Returns
    -------
    An np array of the position on the inlet
    """

    # Assume inlet is vertical
    inlet = data[inlet_set[0],:]
    ymax = np.max(inlet[:,1], axis=0)
    ymin = np.min(inlet[:,1], axis=0)

    # Create Particles
    particles = np.ones((number_particles,2))
    particles[:,0] = particles[:,0] * inlet[0,0]
    particles[:,1] = np.random.uniform(ymin, ymax, 
             size=number_particles)
    particles = particles[particles[:,1].argsort()]
    
    # this may cause problems when number of particles is close
    # to the size of the interval in computer representation
    unique , index = np.unique(particles[:,1], return_index=True)
    if len(unique) != number_particles:
        index_arr = [i for i in range(number_particles)]
        non_unique = [index_arr[i] for i in index_arr if i not in index]
        for i in non_unique:
            particles[i,1] = (particles[i+1,1] - particles[i,1])/2

    return particles


def points_triangle(nt, v, n):
    """
    generates uniform points in nt traingles
    
    Arguements
    ----------
    nt : int
        The number of triangles
    v : array
        A (nt, 3, 2) array of triangle vertices
    n : int
        The number of points in the triangles
    
    Returns
    -------
    r : array
        A (nt, n, 2) array of random points
    """
    
    x = np.sort(np.random.rand(nt ,2, n), axis=1)
    a = np.array([x[:,0], x[:,1] - x[:,0], 1- x[:,1]])
    a = np.transpose(a, (1,2,0))
    z = np.einsum('lij,ljk->lik',a,v)
    return z


def full_interp(geometry, nbt_point_index, num_point=50, 
                kwargs={'method':'linear', 'fill_value':np.nan,
                        'rescale':False}):
    """
    A full interpolation of the whole geomtery
    
    Arguements
    ----------
    geometry : ndarray
        (n,4) the x,y, xvel, yvel from the comsol output
    nbt_point_index : list
        list of triangle vertex indices that do not have all 3
        vertices on the border
    num_points : int
        The number of points in each triangle
    kwargs : dict
        kwargs for scipy.interpolates griddata
        
    returns
    -------
    list of x vel, y vel, interpolation locations
    """
    
    # pass all triangles and generate full uniform points
    point_arr = np.array([geometry[vert,:2] for vert in nbt_point_index])
    interp_points = points_triangle(point_arr.shape[0],
                                    point_arr, num_point)
    fi_points = interp_points.reshape(-1, interp_points.shape[-1])
    
    # interpolation
    interp_x = griddata(geometry[:,:2], geometry[:,2],
                        fi_points, **kwargs)
    interp_y = griddata(geometry[:,:2], geometry[:,3],
                        fi_points, **kwargs)
        
    return [interp_x, interp_y, fi_points]


def find_closest(poi, points_geo):
    """
    Find the point in in points_geo closest to poi
    
    Arguements
    ----------
    poi : ndarray
        (n,2) x,y coordinates of the points of interest
    points_geo : ndarray
        (n,z,2) z number of x,y points for each poi
        
    returns
    -------
    min : ndarray
        points_geo index of minimum distance for each n poi
    """
    
    deltas = poi - points_geo
    dist_2 = np.einsum('nij,nij->ni', deltas, deltas)
    return np.argmin(dist_2, axis=1)


def centroid(triangles):
    """
    Find the centroid of an array of triangles
    
    Arguements
    ----------
    triangles : ndarray
        (n,3,2) n triangles, 3 vertices x,y coordinates
        
    return
    ------
    (n,2) ndarray of x,y coordinates
    """
    
    x = np.sum(triangles[:,0,:], axis=1)/3
    y = np.sum(triangles[:,1,:], axis=1)/3
    return np.array([x,y]).T


def sort_triangles(triangles, method='centroid'):
    """
    Sort the triangles by method
    
    Arguements
    ----------
    triangles : ndarray
        (n,3,2) n triangles, 3 vertices x,y coordinates
    method : str
        either centroid or furthest. centoid measures from centroid
        furthest measures from furthest point from origin
        
    return
    ------
    sorted_t : ndarray
        (n,3,2) array or triangles sorted in increasing order
    """
    
    if method == 'centroid':
        centroids = centroid(triangles)
        point = find_closest(np.array([0,0]), centroids)
        deltas = centroids - centroids[point,:]
        dist = np.einsum('ij,ij->i', deltas, deltas)
    elif method == 'furthest':
        dist = np.max(np.einsum('ijk,ijk->ik', triangles, 
                                triangles), axis=1)
    else:
        raise ValueError('method must be either centroid or furthest')
    sort = np.argsort(dist)
    sorted_t = triangles[sort,:]
    
    return sorted_t


def build_lin_sys(particle_location, triangles):
    """
    Build, solve linear basis transformation and return indicies where
    points are in the triangle
    
    Arguements
    ----------
    particle_location : ndarray
        location of the paricles (n,2) (n,(x,y))
    triangles : ndarray
        location of triangle vertices (n,2,3) (n, (x,y), 3 vertices)
        
    returns
    -------
    tri_location : ndarray
        (x,) indices of particles in the triangle location
    """
    # Create triangular coordinates
    j_ba = triangles[:,:,2] - triangles[:,:,0]
    i_ca = triangles[:,:,1] - triangles[:,:,0]
    transform = np.stack((j_ba,i_ca), axis=2)
    
    # create block diagonal matrix
    system = csr_matrix(block_diag(*transform))
    # Create b vector of the points
    b = (particle_location - triangles[:,:,0]).ravel()
    
    # solve system
    sol = spsolve(system, b)
    
    # basis function test
    n1 = 1 - sol[::2] - sol[1::2]
    n2 = sol[::2]
    n3 = sol[1::2]
    nmat = np.array([n1, n2, n3])
    
    tri_location =np.where(np.logical_and(np.max(nmat, axis=0) <=1 ,
                                          np.max(nmat, axis=0) >=0))
    
    return tri_location[0]


def arb_gen(lst, start):
    """
    Iterate through list from arbitrary starting point
    
    Arguements
    ----------
    lst : List
        List to iterate through
    start : int
        position to start at
        
    Generator
    """
    
    for idx in range(len(lst)):
        yield lst[(idx + start) % len(lst)]

    
def multi_lin_search(particles, sort_triangles):
    """
    Multi linear search: n particles are searched for in j sorted 
    triangles. This search will be faster if the the flow is from left
    to right at the majority of points
    
    Arguements
    ----------
    particles : n,2 or n,3 ndarray
        The x,y, location coordinates of n particles
    sort_triangles : list
        A list of lists of sorted triangle vertices, shape of an nd array
        would be num_triangles,2,3
        
    return
    ------
    p_index : ndarray
        (n,3) (n,(particle index, bool if found, triangle)
    """
    
    num_part = particles.shape[0]
    
    # index, bool if found, triangle location
    p_index = np.array([[i,0,0] for i in range(num_part)])
    
    # create generator 
    if particles.shape[1] == 2:
        search_list = [(tri for tri in arb_gen(sort_triangles, 0)) for 
                       i in range(particles.shape[0])]
    else:
        search_list = [(tri for tri in arb_gen(sort_triangles, i)) for 
                         i in particles[:,2].tolist()]
    
    # tuple of particles yet to be found
    not_found = np.where(p_index[:,1] == 0)
    # main search loop
    for i in range(len(sort_triangles)):
        # particles needed to be found and triangle lists to search
        
        particles_to_pass = particles[not_found,:]
        pass_triangles = [search_list[u] for u in not_found[0].tolist()]
        
        # Get the triangles to seach this iteration
        tri_to_search = np.array([next(tri) for tri in pass_triangles])
        
        # the index of the particles found in triangles and the triangles
        p_indices_found = build_lin_sys(particles_to_pass, tri_to_search)
        indices_to_update = particles_to_pass[p_indices_found[:,0],0]
        p_index[indices_to_update,1] = 1
        p_index[indices_to_update,2] = np.array([tri_to_search[i] for
                                                i in indices_to_update])
        
        # particles not yet found
        not_found = np.where(p_index[:,1] == 0)
        
        if len(not_found) == 0:
            break
        
    return p_index
        

def multi_bin_search(particles, sort_triangles, tri_dist):
    """
    A multiple binary search for particle location. This is the more
    general algorithm and will be faster in all instances except when
    flow is always left to right

    Arguements
    ----------
    particles : n,2 or n,3 ndarray
        The x,y, location coordinates of n particles
    sort_triangles : list
        A list of lists of sorted triangle vertices, shape of an nd array
        would be num_triangles,2,3. Sort should be max distance of 
        vertice from origin
        
    return
    ------
    p_index : ndarray
        (n,3) (n,(particle index, bool if found, triangle)
    """
    
    # index, low, high, triangle location
    num_part = particles.shape[0]
    p_index = np.array([[i,0,0,0] for i in range(num_part)])
    
    # update search array
    low = np.array([0 for i in range(num_part)]) 
    high = np.array([num_part for i in range(num_part)])
    p_index[:,1] = low
    p_index[:,2] = high
    
    while np.any(p_index[:,1] <= p_index[:,2]):
        
        mid_index = p_index[:,1] + (p_index[:,2] - 1) // 2
        
        # get points found array
        p_indices_found = build_lin_sys(particles[mid_index,:],
                                        sort_triangles[mid_index])
        
        p_index[p_indices_found, 1] = mid_index[p_indices_found] + 1
        p_index[p_indices_found, 2:] = mid_index[p_indices_found]
    
        # elements not found
        not_found = np.where(p_index[:,1] == 0)
        
        # particle greater than mid
        greater = np.any(tri_dist[not_found] < particles[not_found])
        p_index[greater,1] = mid_index[0] + 1
        
        # particle less than mid
        less = np.any(tri_dist[not_found] > particles[not_found])
        p_index[less,2] = mid_index[0] - 1
        
    return p_index


def mpl_contains_points(boundary, particles):
    """
    A simple matplotlib.path Path object contains_points
    wrapper. 
    
    Arguements
    ----------
    boundary : set
        the set of x,y boundary points
    particles : ndarray
        (n,2) n x,y coordinates of particles
        
    returns
    -------
    truth_array : ndarray
        bool: True if point is in boundary
    """
    
    poly = Path(boundary)
    truth_array = poly.contains_points(particles)
    return truth_array

    
def simulate(particles, polys, triangles, time_step, interpx, interpy, 
             border_triangles, output='array',output_args =[], 
             max_iter=10000, vel_tol=0.001):
    """
    Simulate the particles trajectory
    
    Arguements
    ----------
    particles : ndarray
        (n,2) array of particle locations
    polys : list
        of path objects defining the outlet polygons
    triangles : Delaunay triangluation object
        The non bonder triangles
    time_step : float
        time interval
    interpx : interpolator object
        callable to interpolate the particles velocity 
    border_triangles : list
        list of triangles which all points line on the border
    output : str
        the output option
    output_args : str
        list of arguements needed for each output option
    max_iter : int
        max number of iterations
    vel_tol : float
        minimum velocity a particle can travel, mostly to deal with
        being stuck on no slip boundary
        
    return
    ------
    array with time, x, y, velx, vely, of each particle
    if output_args = 'array'. If count it returns a count of the 
    number of particles in each outlet. If plot 
    returns a plot of the particles position.
    """
    
    ########################## Initialzation #############################
    
    # initialize array structure with time, x, y, velx, vely, triangle,
    num_part = particles.shape[0]
    tracer_array = np.empty((num_part, max_iter, 6))
    
    # add times
    time_list = np.array([t * time_step for t in range(0, max_iter)])
    tracer_array[:,:,0] = time_list
    
    # add first particles
    tracer_array[:,0,1:3] = particles
    
    bt_index = np.where(border_triangles)
    
    # iteration count
    i = 0
    
    # first velocity
    tracer_array[:,i,3] = interpx(tracer_array[:,i,1:3])    
    tracer_array[:,i,4] = interpy(tracer_array[:,i,1:3])
    
    # stopping conditions for outlets
    tracer_array[:,i,5] = triangles.find_simplex(tracer_array[:,i,1:3])
    out_truth = [poly.contains_points(tracer_array[:,i,1:3]) for 
                        poly in polys]
    outlet_condition = np.where(np.any(np.array(out_truth), axis=0))
    vel_condition = np.where(np.linalg.norm(tracer_array[:,i,3:5], 
                                            axis=1)< vel_tol)[0]
    
    # indexing arrays
    trace = [i for i in range(num_part) if i not in outlet_condition or 
             vel_condition]
    stop = []
    
    # loop
    while len(trace) > 0 and i < max_iter:
        i += 1
        
        # calculate and record position| need to check on this multipilication
        tracer_array[trace,i,1:3] = tracer_array[trace,i-1,1:3] +\
        tracer_array[trace,i-1,3:5] * time_step
        tracer_array[stop,i,1:3] = tracer_array[stop,i-1,1:3]
    
        # Find locations
        tracer_array[trace,i,5] = triangles.find_simplex(
                tracer_array[trace,i,1:3])
        tracer_array[stop,i,5] = tracer_array[stop,i-1,5]
        
        # check if outside the boundary, stop tracing
        out_convex = np.where(tracer_array[:,i,5] == -1)[0]
        part_outside = np.where(np.in1d(tracer_array[:,i,5],
                                        bt_index))[0]
        
        # New velocity
        tracer_array[trace,i,3] = interpx(tracer_array[trace,i,1:3])    
        tracer_array[trace,i,4] = interpy(tracer_array[trace,i,1:3])
        tracer_array[stop,i,3:5] = tracer_array[stop,i-1,3:5]
        
        # update stoping condition
        # stopping conditions for outlets
        out_truth = [poly.contains_points(tracer_array[:,i,1:3]) for 
                        poly in polys]
        outlet_condition = np.where(np.any(np.array(out_truth), 
                                           axis=0))[0]
        vel_condition = np.where(np.linalg.norm(
                tracer_array[:,i,3:5], axis=1) < vel_tol)[0]
        
        # Inidices to stop tracing
        stop = np.unique(np.concatenate((out_convex, part_outside,
                                        outlet_condition, vel_condition)))
        trace = [i for i in trace if i not in stop]    
       
    if out_convex.shape[0] > 0:
        print('particles ' + ', '.join(
                str(x) for x in out_convex.tolist())\
            + ' moved outside convex hull of the geometry.')
        
    if part_outside.shape[0] > 0:
        print('particles ' + ', '.join(
                str(x) for x in part_outside.tolist())\
            + ' moved outside the border geometry.')
    
    # return based on output_args with if statement
    max_iters = i + 1
    
    if output == 'csv':
        output_csv(tracer_array[:,:max_iters,:], *output_args)
    elif output == 'plot':
        plot_trajectories(tracer_array[:,:max_iters,:], *output_args)
    elif output == 'count':
        out_count = np.sum(out_truth, axis=1).tolist()
        outlet_names = ['Outlet ' + str(i) for i in range(len(polys))]
        print('--------------- Results ---------------')
        for name, count in zip(outlet_names, out_count):
            print(name + ' : ' + str(count))
        
    else: 
        return tracer_array[:,:max_iters,:]


##################################################################
# Main
##################################################################

def main():
    
    # Geometry Input and initialization
    args = parser.parse_args()
    if args.file:
        data, names = get_comsol_data(make_file(args.file))
        iodir = pl(args.file).parents[0]
    else:
        iodir = getpath()
        data, names = get_comsol_data(make_file(
                iodir,getfilename('comsol')))
    
    # Use the default arguements for triangles and border
    user_alpha = float(input('Please input a float between 0 and 1 for'\
                             ' the alpha shapes algorithm: '))
    border, triangles = alpha_shape(data[:,:2], alpha=user_alpha)
    
    # Unique Border Points
    ubord = list(set([i for tup in list(border) for i in tup]))
    
    # Delauney triangle index
    point_index = triangles.vertices.tolist()
    
    # border triangles
    bt = [all(elem in ubord for elem in point) for point in point_index]
    
    # Get inlets and outlets
    if args.inlet_outlet:
        inlet_l, outlet_l = in_out_read(make_file(args.inlet_outlet))
    else:
        inlet_l, outlet_l = in_out_read(
                make_file(iodir,getfilename('inlet/outlet')))
    
    inlet, outlet = inlet_outlet(ubord, [inlet_l], outlet_l)
    
    # polygons
    polygons = []
    for out in outlet:
        ymax = np.max(data[out,1], axis=0)
        ymin = np.min(data[out,1], axis=0)
        xmax = np.max(data[out,0], axis=0)
        xmin = xmax - 0.2
        poly = Path([[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]])
        polygons.append(poly)
    
    # choose time step
    if args.timestep:
        timestep = args.timestep
    else:
        timestep = float(input('Please input the time step in seconds: '))


    # get particle inputs and initialize
    if args.num:
        num_particles = args.num
    else:
        num_particles = int(input('Please input the number of'+\
                ' particles: '))
    
    particles = generate_particles(inlet, data[:,:2], num_particles)
    
    
    # get interpolation arguements
    if args.interpolate:
        interpolate = args.interpolate
    else :
        interpolate = input('Please specify the interpolating scheme, '\
                            + 'choices = linear, cubic, nearest. Linear'\
                            + ' is the default ')
        
    if interpolate == 'nearest':
        interpx = NearestNDInterpolator(triangles,data[:,2]*1000)
        interpy = NearestNDInterpolator(triangles,data[:,3]*1000)      
    elif interpolate == 'cubic':
        interpx = CloughTocher2DInterpolator(triangles,data[:,2]*1000)
        interpy = CloughTocher2DInterpolator(triangles,data[:,3]*1000)
    else:
        interpx = LinearNDInterpolator(triangles,data[:,2]*1000)
        interpy = LinearNDInterpolator(triangles,data[:,3]*1000)
    
    # Get output option
    if args.output:
        o_scheme = args.output
    else:
        o_scheme = input('Please specify the output scheme, '\
                         + 'choices = csv, plot, count. Count'\
                         + ' is the default ')
    
    if o_scheme == 'csv':
        iodir = pl(os.getcwd)
        out_args = [iodir]
    elif o_scheme == 'plot':
        out_args = [(15,12), data, border]
    else:
        out_args = []
    
    # Start the simulation
    simulate(particles, polygons, triangles, timestep, interpx, interpy, 
             bt, o_scheme, out_args) 


if __name__ == '__main__':
    main()
