from __future__ import division, absolute_import, print_function
from .common import Benchmark, measure

import sys
from scipy.spatial import cKDTree, KDTree
import numpy as np


class Basics(Benchmark):
    @classmethod
    def gen_build(cls):
        def track_build(self, m, n, cls):
            """
              Constructing kd-tree
            =======================
             dim | # points |  time
            """
            cls = KDTree if cls == 'KDTree' else cKDTree
            m = int(m)
            n = int(n)

            data = np.concatenate((np.random.randn(n//2,m),
                                   np.random.randn(n-n//2,m)+np.ones(m)))

            return measure('cls(data)')

        track_build.unit = "s"

        for (m, n, r) in [(3,10000,1000),
                          (8,10000,1000),
                          (16,10000,1000)]:
            for cls in ('KDTree', 'cKDTree'):
                yield track_build, str(m), str(n), cls

    @classmethod
    def gen_query(cls):
        def track_query(self, m, n, r, cls_str):
            """
            Querying kd-tree
            dim | # points | # queries |  KDTree  | cKDTree | flat cKDTree
            """
            cls = KDTree if cls_str == 'KDTree' else cKDTree
            m = int(m)
            n = int(n)
            r = int(r)

            data = np.concatenate((np.random.randn(n//2,m),
                                   np.random.randn(n-n//2,m)+np.ones(m)))
            queries = np.concatenate((np.random.randn(r//2,m),
                                      np.random.randn(r-r//2,m)+np.ones(m)))

            if cls_str == 'cKDTree_flat':
                T = cls(data, leafsize=n)
            else:
                T = cls(data)

            return measure('T.query(queries)')

        track_query.unit = "s"

        for (m, n, r) in [(3,10000,1000),
                          (8,10000,1000),
                          (16,10000,1000)]:
            for cls in ('KDTree', 'cKDTree', 'cKDTree_flat'):
                yield track_query, str(m), str(n), str(r), cls


    @classmethod
    def gen_query_ball_point(cls):
        def track_query_ball_point(self, m, n, r, probe_radius, cls_str):
            """
            Query ball point kd-tree
            dim | # points | # queries | probe radius |  KDTree  | cKDTree | flat cKDTree
            """
            cls = KDTree if cls_str == 'KDTree' else cKDTree
            m = int(m)
            n = int(n)
            r = int(r)
            probe_radius = float(probe_radius.replace('p', '.'))

            data = np.concatenate((np.random.randn(n//2,m),
                                   np.random.randn(n-n//2,m)+np.ones(m)))
            queries = np.concatenate((np.random.randn(r//2,m),
                                      np.random.randn(r-r//2,m)+np.ones(m)))

            if cls_str == 'cKDTree_flat':
                T = cls(data, leafsize=n)
            else:
                T = cls(data)

            return measure('T.query_ball_point(queries, probe_radius)')

        track_query_ball_point.unit = "s"

        for (m, n, r, repeat) in [(3,10000,1000,3)]:
            for probe_radius in ('0p2', '0p5'):
                for cls in ('KDTree', 'cKDTree', 'cKDTree_flat'):
                    yield track_query_ball_point, str(m), str(n), str(r), probe_radius, cls


    @classmethod
    def gen_query_pairs(cls):
        def track_query_pairs(self, m, n, probe_radius, cls_str):
            """
            Query pairs kd-tree
            dim | # points | probe radius |  KDTree  | cKDTree | flat cKDTree
            """
            cls = KDTree if cls_str == 'KDTree' else cKDTree
            m = int(m)
            n = int(n)
            probe_radius = float(probe_radius.replace('p', '.'))

            data = np.concatenate((np.random.randn(n//2,m),
                                   np.random.randn(n-n//2,m)+np.ones(m)))

            if cls_str == 'cKDTree_flat':
                T = cls(data, leafsize=n)
            else:
                T = cls(data)

            return measure('T.query_pairs(probe_radius)')

        for (m, n, repeat) in [(3,1000,30),
                               (8,1000,30),
                               (16,1000,30)]:
            for probe_radius in ("0p2", "0p5"):
                for cls in ('KDTree', 'cKDTree', 'cKDTree_flat'):
                    yield track_query_pairs, str(m), str(n), probe_radius, cls

    @classmethod
    def gen_sparse_distance_matrix(self):
        def track_sparse_distance_matrix(self, m, n1, n2, probe_radius, cls_str):
            """
            Sparse distance matrix kd-tree
            dim | # points T1 | # points T2 | probe radius |  KDTree  | cKDTree
            """
            cls = KDTree if cls_str == 'KDTree' else cKDTree
            m = int(m)
            n1 = int(n1)
            n2 = int(n2)
            probe_radius = float(probe_radius.replace('p', '.'))

            data1 = np.concatenate((np.random.randn(n1//2,m),
                                    np.random.randn(n1-n1//2,m)+np.ones(m)))
            data2 = np.concatenate((np.random.randn(n2//2,m),
                                    np.random.randn(n2-n2//2,m)+np.ones(m)))

            T1 = cls(data1)
            T2 = cls(data2)

            return measure('T1.sparse_distance_matrix(T2, probe_radius)')

        track_sparse_distance_matrix.unit = "s"

        for (m, n1, n2, repeat) in [(3,1000,1000,30),
                                    (8,1000,1000,30),
                                    (16,1000,1000,30)]:
            for probe_radius in ("0p2", "0p5"):
                for cls in ('KDTree', 'cKDTree'):
                    yield track_sparse_distance_matrix, str(m), str(n1), str(n2), probe_radius, cls

    @classmethod
    def gen_count_neighbors(self):
        def track_count_neighbors(self, m, n1, n2, probe_radius, cls_str):
            """
            Count neighbors kd-tree
            dim | # points T1 | # points T2 | probe radius |  KDTree  | cKDTree
            """

            cls = KDTree if cls_str == 'KDTree' else cKDTree
            m = int(m)
            n1 = int(n1)
            n2 = int(n2)
            probe_radius = float(probe_radius.replace('p', '.'))

            data1 = np.concatenate((np.random.randn(n1//2,m),
                                    np.random.randn(n1-n1//2,m)+np.ones(m)))
            data2 = np.concatenate((np.random.randn(n2//2,m),
                                    np.random.randn(n2-n2//2,m)+np.ones(m)))

            T1 = KDTree(data1)
            T2 = KDTree(data2)
            cT1 = cKDTree(data1)
            cT2 = cKDTree(data2)

            return measure('T1.count_neighbors(T2, probe_radius)')

        track_count_neighbors.unit = "s"

        for (m, n1, n2, repeat) in [(3,1000,1000,30),
                                    (8,1000,1000,30),
                                    (16,1000,1000,30)]:
            for probe_radius in ("0p2", "0p5"):
                for cls in ('KDTree', 'cKDTree'):
                    yield track_count_neighbors, str(m), str(n1), str(n2), probe_radius, cls
