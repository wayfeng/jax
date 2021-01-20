# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# flake8: noqa

from contextlib import contextmanager
import functools
import itertools as it
import os
import unittest
from typing import (Tuple, List, NamedTuple, Dict, Generator, Sequence, Set,
                    Any, Hashable, Iterable, Iterator)
from unittest import SkipTest, skip, skipIf

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from functools import partial

import jax
import jax.numpy as jnp
from jax import test_util as jtu
from jax import vmap
from jax import lax
from jax.experimental.maps import Mesh, mesh, xmap
from jax.lib import xla_bridge
from jax._src.util import curry, unzip2, split_list, prod
from jax._src.lax.lax import DotDimensionNumbers
from jax.interpreters import pxla

from jax.config import config
config.parse_flags_with_absl()

ignore_xmap_warning = functools.partial(
  jtu.ignore_warning, message=".*is an experimental.*")

# TODO(mattjj): de-duplicate setUpModule and tearDownModule with pmap_test.py
# Run all tests with 8 CPU devices.
def setUpModule():
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=8")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


@curry
def with_mesh(named_shape, f):
  if not named_shape:
    return f
  def new_f(*args, **kwargs):
    axis_names, shape = unzip2(named_shape)
    size = np.prod(shape)
    local_devices = list(jax.local_devices())
    if len(local_devices) < size:
      raise SkipTest(f"Test requires {size} local devices")
    mesh_devices = np.array(local_devices[:size]).reshape(shape)
    with mesh(mesh_devices, axis_names):
      return f(*args, **kwargs)
  return new_f


class XMapTest(jtu.JaxTestCase):
  def setUp(self):
    if jax.lib.version < (0, 1, 58):
      raise SkipTest("xmap requires jaxlib version >= 0.1.58")
    if not config.omnistaging_enabled:
      raise SkipTest("xmap requires omnistaging")

  @ignore_xmap_warning()
  def testBasic(self):
    local_devices = list(jax.local_devices())
    if len(local_devices) < 4:
      raise SkipTest("Test requires at least 4 local devices")
    def f(a, b):
      return a * 2, b * 4
    devices = np.array(local_devices[:4]).reshape((2, 2))
    with mesh(devices, ('x', 'y')):
      fm = xmap(f,
                in_axes=[{0: 'a', 1: 'b'}, ['c', ...]],
                out_axes=[{0: 'a', 1: 'b'}, ['c', ...]],
                axis_resources={'a': 'x', 'b': 'y', 'c': 'x'})
      ashape = (16, 8, 5)
      a = jnp.arange(np.prod(ashape)).reshape(ashape)
      bshape = (2, 7)
      b = jnp.arange(np.prod(bshape)).reshape(bshape)
      c, d = fm(a, b)
      self.assertAllClose(c, a * 2)
      self.assertAllClose(d, b * 4)

  @ignore_xmap_warning()
  def testBasicCollective(self):
    local_devices = list(jax.local_devices())
    if len(local_devices) < 4:
      raise SkipTest("Test requires at least 4 local devices")
    def f(a, b):
      return lax.psum(a * 2, 'a'), b * 4
    devices = np.array(local_devices[:4]).reshape((2, 2))
    with mesh(devices, ('x', 'y')):
      fm = xmap(f,
                in_axes=[['a', 'b', ...], {0: 'c'}],
                out_axes=[['b', ...], {0: 'c'}],
                axis_resources={'a': 'x', 'b': 'y', 'c': 'x'})
      ashape = (16, 8, 5)
      a = jnp.arange(np.prod(ashape)).reshape(ashape)
      bshape = (2, 7)
      b = jnp.arange(np.prod(bshape)).reshape(bshape)
      c, d = fm(a, b)
      self.assertAllClose(c, (a * 2).sum(0))
      self.assertAllClose(d, b * 4)

  @ignore_xmap_warning()
  @with_mesh([('x', 2), ('y', 2)])
  def testOneLogicalTwoMeshAxesBasic(self):
    def f(v):
      return lax.psum(v * 2, 'a'), v * 4
    fm = xmap(f, in_axes=['a', ...], out_axes=[{}, {1: 'a'}],
              axis_resources={'a': ('x', 'y')})
    vshape = (4, 5)
    v = jnp.arange(np.prod(vshape)).reshape(vshape)
    ans, ans2 = fm(v)
    self.assertAllClose(ans, (v * 2).sum(0))
    self.assertAllClose(ans2, v.T * 4)

  @ignore_xmap_warning()
  @with_mesh([('x', 2), ('y', 2)])
  def testOneLogicalTwoMeshAxesSharding(self):
    def f(v):
      return v * 4
    fxy = xmap(f, in_axes=['a', ...], out_axes={1: 'a'},
               axis_resources={'a': ('x', 'y')})
    fyx = xmap(f, in_axes=['a', ...], out_axes={1: 'a'},
               axis_resources={'a': ('y', 'x')})
    vshape = (4, 5)
    v = jnp.arange(np.prod(vshape)).reshape(vshape)
    zxy = fxy(v)
    self.assertEqual(
        zxy.sharding_spec,
        pxla.ShardingSpec((None, pxla.Chunked((2, 2))),
                          (pxla.ShardedAxis(0), pxla.ShardedAxis(1))))
    zyx = fyx(v)
    self.assertEqual(
        zyx.sharding_spec,
        pxla.ShardingSpec((None, pxla.Chunked((2, 2))),
                          (pxla.ShardedAxis(1), pxla.ShardedAxis(0))))

  @ignore_xmap_warning()
  @with_mesh([('x', 2)])
  def testCompilationCache(self):
    def f(x):
      assert python_should_be_executing
      return x * 2
    fm = xmap(f,
              in_axes=['a', ...], out_axes=['a', ...],
              axis_resources={'a': 'x'})
    x = np.arange(8).reshape((2, 2, 2))
    python_should_be_executing = True
    fm(x)
    python_should_be_executing = False
    fm(x)

  @skip("Need to implement vmap(xmap)")
  @ignore_xmap_warning()
  @with_mesh([('x', 2)])
  def testNestedVectorize(self):
    @partial(xmap, in_axes=[None, 'a', ...], out_axes=['a', ...], axis_resources={'a': 'x'})
    def f(x):
      y = x * 2
      @partial(xmap, in_axes=['b', ...], out_axes=[None, 'b', ...])
      def h(y):
        return jnp.sin(y)
      return h(y)
    xshape = (4, 2, 5)
    x = jnp.arange(np.prod(xshape)).reshape(xshape)
    self.assertAllClose(f(x),
                        jnp.sin(x * 2).transpose((1, 2, 0)))

  @skip("Need to implement vmap(xmap)")
  @ignore_xmap_warning()
  @with_mesh([('x', 2), ('y', 3)])
  def testNestedMesh(self):
    @partial(xmap, in_axes={1: 'a'}, out_axes={0: 'a'}, axis_resources={'a': 'y'})
    def f(x):
      y = x * 2
      @partial(xmap, in_axes={0: 'b'}, out_axes={1: 'b'}, axis_resources={'b': 'x'})
      def h(y):
        return jnp.sin(y)
      return h(y)
    xshape = (2, 3, 5)
    x = jnp.arange(np.prod(xshape)).reshape(xshape)
    y = f(x)
    self.assertAllClose(y, jnp.sin(x * 2).transpose((1, 2, 0)))
    # Make sure the op really ran accros a 2D mesh.
    self.assertEqual(y.sharding_spec.sharding,
                     (pxla.Chunked(3), None, None))
    self.assertEqual(y.sharding_spec.mesh_mapping,
                     (pxla.Replicated(2), pxla.ShardedAxis(0)))

  @ignore_xmap_warning()
  @with_mesh([('x', 2)])
  def testNestedDifferentResources(self):
    @partial(xmap, in_axes={0: 'a'}, out_axes={0: 'a'}, axis_resources={'a': 'x'})
    def f(x):
      with mesh(np.empty((), dtype=np.object), ()):
        @partial(xmap, in_axes={0: 'b'}, out_axes={0: 'b'})
        def h(x):
          return x
        return h(x)
    xshape = (2, 5, 6)
    x = jnp.arange(np.prod(xshape)).reshape(xshape)
    with self.assertRaisesRegex(RuntimeError,
                                "Changing the resource environment.*"):
      f(x)

  @parameterized.named_parameters(
    {"testcase_name": name, "mesh": mesh, "axis_resources": axis_resources}
    for name, mesh, axis_resources in (
      ('', (), ()),
      ('Mesh', (('x', 2),), (('i', 'x'),))
    ))
  @ignore_xmap_warning()
  def testMultipleCalls(self, mesh, axis_resources):
    def f(x, y):
      assert x.shape == y.shape == (3, 5)
      return jnp.tensordot(x, y, axes=([1], [1]))

    @with_mesh(mesh)
    def run_test():
      f_mapped = xmap(f,
                      in_axes=(['i', ...], ['j', ...]),
                      out_axes=['i', 'j', ...],
                      axis_resources=dict(axis_resources))
      x = jnp.arange(30).reshape(2, 3, 5)
      expected = jnp.einsum('imk,jnk->ijmn', x, x)
      for i in range(10):
        self.assertAllClose(f_mapped(x, x), expected)
    run_test()


class XMapTestSPMD(XMapTest):
  """Re-executes all tests with the SPMD partitioner enabled"""

  def setUp(self):
    super().setUp()
    if jtu.device_under_test() != "tpu":
      raise SkipTest
    jax.experimental.maps.make_xmap_callable.cache_clear()
    self.old_lowering_flag = jax.experimental.maps.EXPERIMENTAL_SPMD_LOWERING
    jax.experimental.maps.EXPERIMENTAL_SPMD_LOWERING = True

  def tearDown(self):
    jax.experimental.maps.make_xmap_callable.cache_clear()
    jax.experimental.maps.EXPERIMENTAL_SPMD_LOWERING = self.old_lowering_flag

AxisIndices = Tuple[int, ...]
MatchedAxisIndices = Tuple[AxisIndices, AxisIndices]
AxisNames = Tuple[str, ...]

class PdotTestSpec:
  # The axis indices stored by a PdotTestSpec are all positional indices
  # *before* taking mapping into account.
  map_cont: MatchedAxisIndices
  pos_cont: MatchedAxisIndices
  map_batch: MatchedAxisIndices
  pos_batch: MatchedAxisIndices
  all_names: AxisNames
  contract_names: AxisNames
  batch_names: AxisNames

  def __init__(self, map_cont, pos_cont, map_batch, pos_batch):
    self.map_cont = map_cont
    self.pos_cont = pos_cont
    self.map_batch = map_batch
    self.pos_batch = pos_batch

    names = gen_axis_names()
    self.contract_names = [next(names) for _ in range(len(map_cont[0]))]
    self.batch_names = [next(names) for _ in range(len(map_batch[0]))]
    self.all_names = self.contract_names + self.batch_names

  @property
  def dot_general_dim_nums(self):
    lhs_contract = (*self.map_cont[0], *self.pos_cont[0])
    rhs_contract = (*self.map_cont[1], *self.pos_cont[1])
    lhs_batch = (*self.map_batch[0], *self.pos_batch[0])
    rhs_batch = (*self.map_batch[1], *self.pos_batch[1])
    return (lhs_contract, rhs_contract), (lhs_batch, rhs_batch)

  @property
  def pos_contract_after_mapping(self):
    lhs = [i - sum(j < i for j in self._lhs_mapped) for i in self.pos_cont[0]]
    rhs = [i - sum(j < i for j in self._rhs_mapped) for i in self.pos_cont[1]]
    return (lhs, rhs)

  @property
  def pos_batch_after_mapping(self):
    lhs = [i - sum(j < i for j in self._lhs_mapped) for i in self.pos_batch[0]]
    rhs = [i - sum(j < i for j in self._rhs_mapped) for i in self.pos_batch[1]]
    return (lhs, rhs)

  @property
  def _lhs_mapped(self):
    return {*self.map_cont[0], *self.map_batch[0]}

  @property
  def _rhs_mapped(self):
    return {*self.map_cont[1], *self.map_batch[1]}

  @property
  def lhs_in_axes(self):
    axis_indices = [*self.map_cont[0], *self.map_batch[0]]
    return dict(zip(axis_indices, self.all_names))

  @property
  def rhs_in_axes(self):
    axis_indices = [*self.map_cont[1], *self.map_batch[1]]
    return dict(zip(axis_indices, self.all_names))

def all_pdot_specs(lhs_shape, rhs_shape):
  for matching in axis_matchings(lhs_shape, rhs_shape):
    for lists in partitions(matching, 4):
      yield PdotTestSpec(*map(unzip2, lists))

def axis_matchings(lhs_shape, rhs_shape):
  def helper(start, exc1, exc2):
    yield ()
    for i in range(start, len(lhs_shape)):
      d1 = lhs_shape[i]
      if i not in exc1:
        for j, d2 in enumerate(rhs_shape):
          if d1 == d2 and j not in exc2:
            for matches in helper(i + 1, exc1 | {i}, exc2 | {j}):
              yield ((i, j), *matches)
  return helper(0, set(), set())

def partitions(s, k):
  for indices in it.product(range(k), repeat=len(s)):
    outs = [[] for _ in range(k)]
    for i, elt in zip(indices, s):
      outs[i].append(elt)
    yield outs

def powerset(s):
  s = list(s)
  return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))

def gen_axis_names():
  names = 'ijkl'
  for n in it.count(1):
    for chars in it.product(names, repeat=n):
      yield ''.join(chars)

AxisResources = List[Dict[str, str]]
MeshSpec = List[Tuple[str, int]]

def schedules(sizes: Dict[str, int]
              ) -> Generator[Tuple[AxisResources, MeshSpec], None, None]:
  """Test utility generating xmap parallel schedules from logical names & sizes.

  Args:
    sizes: dict mapping logical axis name to its corresponding size.

  Returns:
    A generator producing finitely many values, where each value is a pair in
    which the first element is a value suitable for xmap's axis_resources
    argument and the second element is a list of pairs with the first element
    representing a generated physical mesh axis name and the second element
    representing a corresponding generated mesh axis size. The generated mesh
    names/sizes can be used to define a physical mesh in tests.

  This function doesn't generate schedules which map distinct logical axis names
  to the same parallel resource name. It only generates parallel resources; the
  rest are implicitly left for vectorization. Parallel resource names are
  generated by prepending an 'r' to the corresponding logical name.

  Examples:
    >>> for sched in schedules({'i': 2, 'j': 4}):
    ...   print(sched)
    ([], [])
    ([('i', 'ri')], [('ri', 1)])
    ([('i', 'ri')], [('ri', 2)])
    ([('j', 'rj')], [('rj', 1)])
    ([('j', 'rj')], [('rj', 2)])
    ([('j', 'rj')], [('rj', 4)])
    ([('j', 'rj'), ('i', 'ri')], [('rj', 1), ('ri', 1)])
    ([('j', 'rj'), ('i', 'ri')], [('rj', 1), ('ri', 2)])
    ([('j', 'rj'), ('i', 'ri')], [('rj', 2), ('ri', 1)])
    ([('j', 'rj'), ('i', 'ri')], [('rj', 2), ('ri', 2)])
    ([('j', 'rj'), ('i', 'ri')], [('rj', 4), ('ri', 1)])
    ([('j', 'rj'), ('i', 'ri')], [('rj', 4), ('ri', 2)])
  """
  def divisors(n: int) -> List[int]:
    return [m for m in range(1, n + 1) if not n % m]

  for names in powerset(sizes):
    for mesh_sizes in it.product(*(divisors(sizes[n]) for n in names)):
      axis_resources = dict((name, 'r' + name) for name in names)
      mesh_data = [('r' + name, size) for name, size in zip(names, mesh_sizes)]
      yield axis_resources, mesh_data

@contextmanager
def generate_mesh(named_shape: MeshSpec) -> Generator[None, None, None]:
  """Test utility for setting up meshes given mesh data from `schedules`."""
  # This is similar to the `with_mesh` function above, but isn't a decorator.
  # Could de-duplicate.
  axis_names, shape = unzip2(named_shape)
  size = prod(shape)
  local_devices = list(jax.local_devices())
  if len(local_devices) < size:
    raise SkipTest(f"Test requires {size} local devices")
  mesh_devices = np.array(local_devices[:size]).reshape(shape)
  with mesh(mesh_devices, axis_names):
    yield

def schedules_from_pdot_spec(
    spec: PdotTestSpec, lhs_shape: Tuple[int], rhs_shape: Tuple[int]
    ) -> Generator[Tuple[AxisResources, MeshSpec], None, None]:
  logical_sizes = {
      name: shape[ax]
      for shape, in_axes in [(lhs_shape, spec.lhs_in_axes),
                                 (rhs_shape, spec.rhs_in_axes)]
      for ax, name in in_axes.items()}
  yield from schedules(logical_sizes)


class PDotTests(jtu.JaxTestCase):

  def setUp(self):
    if not config.omnistaging_enabled:
      raise SkipTest("xmap requires omnistaging")
    super().setUp()

  @ignore_xmap_warning()
  @with_mesh([('r1', 2)])
  def testPdotBasic(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    f_mapped = xmap(f,
                    in_axes=[{1: 'i'}, {0: 'i'}],
                    out_axes={},
                    axis_resources={'i': 'r1'})

    rng = np.random.RandomState(0)
    x = rng.randn(3, 8)
    y = rng.randn(8, 5)

    z = f_mapped(x, y)

    self.assertAllClose(z, jnp.dot(x, y))

  @ignore_xmap_warning()
  @with_mesh([('r1', 2)])
  def testPdotBatching(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    rng = np.random.RandomState(0)
    x = rng.randn(2, 3, 8)
    y = rng.randn(2, 8, 5)

    f_mapped = xmap(f,
                    in_axes=[{0: 'j', 2: 'i'}, {0: 'j', 1: 'i'}],
                    out_axes=['j', ...],
                    axis_resources={'i': 'r1'})

    z = f_mapped(x, y)

    self.assertAllClose(z, jnp.einsum('nij,njk->nik', x, y))

  @ignore_xmap_warning()
  @with_mesh([('r1', 2)])
  def testPdotBatchingShardUncontractedDim(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    rng = np.random.RandomState(0)
    x = rng.randn(2, 3, 8)
    y = rng.randn(2, 8, 5)

    f_mapped = xmap(f,
                    in_axes=[{0: 'j', 2: 'i'}, {0: 'j', 1: 'i'}],
                    out_axes=['j', ...],
                    axis_resources={'j': 'r1'})

    z = f_mapped(x, y)

    self.assertAllClose(z, jnp.einsum('nij,njk->nik', x, y))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{next(test_counter)}",
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "pdot_spec": pdot_spec,
       "axis_resources": axis_resources, "mesh_data": mesh_data}
      for test_counter in [it.count()]
      for lhs_shape, rhs_shape in it.product(
          [(2, 2), (2, 3, 2), (2, 3, 2, 1)],
          repeat=2)
      for pdot_spec in all_pdot_specs(lhs_shape, rhs_shape)
      for axis_resources, mesh_data in schedules_from_pdot_spec(
          pdot_spec, lhs_shape, rhs_shape)))
  @ignore_xmap_warning()
  def testPdotSystematic(self, lhs_shape, rhs_shape, pdot_spec, axis_resources,
                         mesh_data):
    rng = jtu.rand_default(self.rng())
    lhs = rng(lhs_shape, np.float32)
    rhs = rng(rhs_shape, np.float32)

    def pdot_fun(x, y):
      # print(f'pdot(x:{x.aval.str_short()}, y:{y.aval.str_short()},\n'
      #       f'     axis_name={contract_names},\n'
      #       f'     pos_contract={spec.pos_contract_after_mapping}\n'
      #       f'     pos_batch={spec.pos_batch_after_mapping})')
      return jax.lax.pdot(x, y, axis_name=pdot_spec.contract_names,
                          pos_batch=pdot_spec.pos_batch_after_mapping,
                          pos_contract=pdot_spec.pos_contract_after_mapping)

    fun = xmap(pdot_fun, in_axes=[pdot_spec.lhs_in_axes, pdot_spec.rhs_in_axes],
               out_axes=[*pdot_spec.batch_names, ...],
               axis_resources=axis_resources)

    with generate_mesh(mesh_data):
      result = fun(lhs, rhs)

    expected = lax.dot_general(lhs, rhs, pdot_spec.dot_general_dim_nums)
    self.assertAllClose(result, expected, atol=1e-3, rtol=1e-3)


class XMapErrorTest(jtu.JaxTestCase):

  @ignore_xmap_warning()
  @with_mesh([('x', 2)])
  def testRepeatedAxisResource(self):
    def f(v):
      return v * 4
    with self.assertRaisesRegex(ValueError, r"distinct resources.*specified \('x', 'x'\) for axis a"):
      fxy = xmap(f, in_axes=['a', ...], out_axes=['a', ...],
                 axis_resources={'a': ('x', 'x')})


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
