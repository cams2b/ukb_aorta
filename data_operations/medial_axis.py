from numpy.random import rand
from numpy import array, zeros, concatenate, nan
import numpy as np
from itertools import chain

from scipy.spatial import KDTree


from multiprocessing import Process, Pool, cpu_count, Manager
from multiprocessing.queues import Queue
from sklearn.decomposition import PCA
import math
from numpy.linalg import norm
from time import time



# numpy based operations for any dimension

def equal(vec1, vec2):
    return (vec1==vec2).all()

def dot(vec1, vec2):
    """ Calculate the dot product of two 3d vectors. """
    return vec1.dot(vec2)

def normalize(vec):
    """ Calculate the normalized vector (norm: one). """
    return vec / norm(vec)

def proj(u, v):
    factor = u.dot(v) / u.dot(u)
    return factor*u

def projfac(u, v):
    factor = u.dot(v) / u.dot(u)
    return factor

def cos_angle(p, q):
    return p.dot(q) / ( norm(p) * norm(q) )

def compute_radius(p, p_n, q):
    """compute radius of ball that touches points p and q and is centered on along the normal p_n of p. Numpy array inputs."""
    # this is basic goniometry
    d = norm(p-q)
    cos_theta = p_n.dot(p-q) / d
    return d/(2*cos_theta)
    

class medial_axis(object):
    ## https://github.com/tudelft3d/masbpy
    def __init__(self, data_dict, max_radius, denoise=None, denoise_delta=None, detect_planar=None):

        #self.manager = Manager()
        self.data = data_dict # numpy arrays
        self.m, self.n = data_dict['coords'].shape

        self.kd_tree = KDTree(self.data['coords'])
        self.SuperR = max_radius
        self.denoise = denoise
        self.denoise_delta = denoise_delta
        self.detect_planar = detect_planar
    
    def compute_sp(self):
        from Queue import Queue
        queue = Queue()
        datalen = len(self.data['coords'])
        self(queue,0,datalen, True, False)
        self(queue,0,datalen, False, False)
        return queue.get() + queue.get()

    def compute_balls(self, num_processes=cpu_count()):
        print('[INFO] computing MAT')
        datalen = len(self.data['coords'])

        n = int(num_processes/2)
        batchsize = datalen/n

        chunk_bounds = []
        end=0
        for i in range(n-1):
            start = end
            end = (i + 1) * batchsize - 1
            chunk_bounds.append( (int(start), int(end)) )
        start, end = end, datalen
        chunk_bounds.append((int(start), int(end)))

        jobs = []
        manager = Manager()
        queue = manager.Queue()

        t1 = time()
        for s, e in chunk_bounds:
            p1 = Process(target=self, args=(queue, s, e, True, False))
            p1.start()
            jobs.append(p1)
            

            p2 = Process(target=self, args=(queue, s, e, False, False))
            p2.start()
            jobs.append(p2)

        result = []
        for j in jobs:
            j.join()
            res = queue.get()
            result.append(res)

        t2 = time



        result.sort(key=lambda item: (item[1], item[0]))

        self.data['ma_coords_out'] = concatenate([ma_coords for start, inner, ma_coords, ma_f2 in result[:n] ])
        self.data['ma_f2_out'] = concatenate([ma_f2 for start, inner, ma_coords, ma_f2 in result[:n] ])
        
        self.data['ma_coords_in'] = concatenate([ma_coords for start, inner, ma_coords, ma_f2 in result[n:] ])
        self.data['ma_f2_in'] = concatenate([ma_f2 for start, inner, ma_coords, ma_f2 in result[n:] ])
        
        


    
    def __call__(self, queue, start, end, inner=True, verbose=False):
        """
        Balls shrinking algorithm. Set inner to False when outer balls are wanted
        """
        print('we get here !!!!')
        m = end - start

        ma_coords = zeros((int(m), int(self.n)), dtype=np.float32)
        ma_coords[:] = nan

        ma_f2 = zeros(int(m), dtype=np.float32)
        ma_f2[:] = nan

        ZeroDivisionError_cnt = 0
        for i, pi, in enumerate(range(start, end)):
            p, n = self.data['coords'][pi], self.data['normals'][pi]

            if not inner:
                n = -n
            
            # use the previous point as the initial estimate for q
            q=p

            r = self.SuperR

            r_ = None
            c = None
            j = -1
            q_i = None
            q_history = []

            while True:
                j += 1

                # initialize r on the last found radius
                if j > 0:
                    r = r_
                elif j == 0 and i > 0:
                    r = r

                
                # compute ball center
                c = p - n*r

                q_i_previous = q_i

                # find closest point to c and assign to q
                dists, results = self.kd_tree.query(array([c]), k=2)
                candidate_c = self.data['coords'][results]

                q = candidate_c[0][0]
                q_i = results[0][0]

                if equal(q, p):
                    if r == self.SuperR: break

                    else:
                        q = candidate_c[0][1]
                        q_i = results[0][1]
                    
                try:
                    r_ = compute_radius(p, n, q)
                except ZeroDivisionError:
                    ZeroDivisionError_cnt += 1
                    r_ = self.SuperR+1
                
                # if r_ < 0 closest point was on the wrong side of plane with normal n => start over with SuperRadius on the right side of that plance
                if r_ < 0. : r_ = self.SuperR

                elif r_ > self.SuperR:
                    r_ = self.SuperR
                    break
                
                c = p - n*r_
                
                if abs(r - r_) < 0.01:
                    break

                if j > 30:
                    break
            
            if r_ == None or r_ >= self.SuperR:
                pass
            else:
                ma_coords[i] = c
                ma_f2[i] = q_i
        
        result = ( start, inner, ma_coords, ma_f2 )
        queue.put(result)

                

            


            

































def compute_normal(neighbours):
    pca = PCA(n_components=3)
    pca.fit(neighbours)
    plane_normal = pca.components_[-1] # this is a normalized normal
	# make all normals 
    if plane_normal[-1] < 0:
        plane_normal *= 1
    return plane_normal



def format_data(data_dict):
    kd_tree = KDTree(data_dict['coords'])


    neighbours = kd_tree.query( data_dict['coords'], k=10 )[1]
    
    
    neighbours = data_dict['coords'][neighbours]
    
    p = Pool()
    normals = p.map(compute_normal, neighbours)
    data_dict['normals'] = np.array(normals, dtype=np.float32)

    return data_dict
    
    







