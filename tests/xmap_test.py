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


LHSSpec = RHSSpec = Tuple[int, ...]
DimSpec = Tuple[LHSSpec, RHSSpec]

class PdotSpec(NamedTuple):
  """Specifies positional axes to contract or batch via mapping or position."""
  map_contract: DimSpec
  pos_contract: DimSpec
  map_batch: DimSpec
  pos_batch: DimSpec

  @property
  def num_mapped(self):
    return len(self.lhs_mapped)

  @property
  def num_mapped_contract(self):
    return len(self.map_contract[0])

  @property
  def lhs_mapped(self):
    return [*self.map_contract[0], *self.map_batch[0]]

  @property
  def rhs_mapped(self):
    return [*self.map_contract[1], *self.map_batch[1]]

  @property
  def pos_contract_after_mapping(self):
    lhs_pos_contract_ = [i - sum(j < i for j in self.lhs_mapped)
                         for i in self.pos_contract[0]]
    rhs_pos_contract_ = [i - sum(j < i for j in self.rhs_mapped)
                         for i in self.pos_contract[1]]
    return (lhs_pos_contract_, rhs_pos_contract_)

  @property
  def pos_batch_after_mapping(self):
    lhs_pos_batch_ = [i - sum(j < i for j in self.lhs_mapped)
                      for i in self.pos_batch[0]]
    rhs_pos_batch_ = [i - sum(j < i for j in self.rhs_mapped)
                      for i in self.pos_batch[1]]
    return (lhs_pos_batch_, rhs_pos_batch_)

  @property
  def all_contract(self):
    lhs_contract = [*self.map_contract[0], *self.pos_contract[0]]
    rhs_contract = [*self.map_contract[1], *self.pos_contract[1]]
    return lhs_contract, rhs_contract

  @property
  def all_batch(self):
    lhs_batch = [*self.map_batch[0], *self.pos_batch[0]]
    rhs_batch = [*self.map_batch[1], *self.pos_batch[1]]
    return lhs_batch, rhs_batch

def all_pdot_specs(lhs_shape: Tuple[int], rhs_shape: Tuple[int]
                   ) -> Generator[PdotSpec, None, None]:
  """Test utility to generate all valid pdot specs for given shapes."""
  for dim_nums in all_dot_general_dimension_numbers(lhs_shape, rhs_shape):
    yield from map_some_axes(dim_nums)

def all_dot_general_dimension_numbers(
    lhs_shape: Sequence[int], rhs_shape: Sequence[int]
  ) -> Generator[DotDimensionNumbers, None, None]:
  """Test utility to generate all valid dot_general specs for given shapes."""
  # The recursive 'helper' function generates all possible DotDimensionNumbers
  # while excluding certain axis positions. The base case in the first yield is
  # not to assign any additional axes to be batched or contracted, while the
  # recursive cases in the latter two yields come from choosing equal-sized axes
  # to batch or contract (recursing to possibly choose more axes).
  def helper(excluded1: Set[int], excluded2: Set[int]):
    yield ((), ()), ((), ())
    for i, d1 in enumerate(lhs_shape):
      if i not in excluded1:
        for j, d2 in enumerate(rhs_shape):
          if j not in excluded2 and d1 == d2 and j <= i:
            for contract, batch in helper(excluded1 | {i}, excluded2 | {j}):
              (lhs_cont, rhs_cont), (lhs_batch, rhs_batch) = contract, batch
              yield (lhs_cont + (i,), rhs_cont + (j,)), batch
              yield contract, (lhs_batch + (i,), rhs_batch + (j,))
  return helper(set(), set())

def map_some_axes(dim_nums: DotDimensionNumbers
                  ) -> Generator[PdotSpec, None, None]:
  """Helper function to generate pdot specs from dot_general dim nums."""
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
  for idx1 in powerset(range(len(lhs_contract))):
    map_lhs_contract, pos_lhs_contract = partition(idx1, lhs_contract)
    map_rhs_contract, pos_rhs_contract = partition(idx1, rhs_contract)
    for idx2 in powerset(range(len(lhs_batch))):
      map_lhs_batch, pos_lhs_batch = partition(idx2, lhs_batch)
      map_rhs_batch, pos_rhs_batch = partition(idx2, rhs_batch)
      yield PdotSpec((map_lhs_contract, map_rhs_contract),
                     (pos_lhs_contract, pos_rhs_contract),
                     (map_lhs_batch, map_rhs_batch),
                     (pos_lhs_batch, pos_rhs_batch))

def powerset(itr: Iterable[Hashable]) -> Iterator[Set[Hashable]]:
  # from an itertools recipe
  lst = list(itr)
  outs = map(set, (it.combinations(lst, r) for r in range(len(lst) + 1)))
  return it.chain.from_iterable(outs)

def partition(mapped_idx: Set[int], lst: List[Any]
              ) -> Tuple[List[Any], List[Any]]:
  lists = [], []
  for i, elt in enumerate(lst):
    lists[i not in mapped_idx].append(elt)
  return lists

def gen_axis_names(n: int):
  def _axis_names():
    names = 'ijkl'
    for n in it.count(1):
      for chars in it.product(names, repeat=n):
        yield ''.join(chars)
  return list(it.islice(_axis_names(), n))

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
    spec: PdotSpec, lhs_shape: Tuple[int], rhs_shape: Tuple[int]
    ) -> Generator[Tuple[AxisResources, MeshSpec], None, None]:
  names = gen_axis_names(spec.num_mapped)
  lhs_spec = dict(zip(spec.lhs_mapped, names))
  rhs_spec = dict(zip(spec.rhs_mapped, names))
  logical_sizes = {
      name: shape[ax]
      for shape, spec in [(lhs_shape, lhs_spec), (rhs_shape, rhs_spec)]
      for ax, name in spec.items()}
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
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "spec": spec,
       "axis_resources": axis_resources, "mesh_data": mesh_data}
      for test_counter in [it.count()]
      for lhs_shape, rhs_shape in it.product(
          [(2, 2), (2, 3, 2), (2, 3, 1, 2)],
          repeat=2)
      for spec in all_pdot_specs(lhs_shape, rhs_shape)
      for axis_resources, mesh_data in schedules_from_pdot_spec(
          spec, lhs_shape, rhs_shape)))
  @ignore_xmap_warning()
  def testPdotSystematic(self, lhs_shape, rhs_shape, spec, axis_resources,
                         mesh_data):
    names = gen_axis_names(spec.num_mapped)
    contract_names, batch_names = split_list(names, [spec.num_mapped_contract])
    lhs_spec = dict(zip(spec.lhs_mapped, names))
    rhs_spec = dict(zip(spec.rhs_mapped, names))

    rng = jtu.rand_default(self.rng())
    lhs = rng(lhs_shape, np.float32)
    rhs = rng(rhs_shape, np.float32)

    def pdot_fun(x, y):
      print(f'pdot(x:{x.aval.str_short()}, y:{y.aval.str_short()},\n'
            f'     axis_name={contract_names},\n'
            f'     pos_contract={spec.pos_contract_after_mapping}\n'
            f'     pos_batch={spec.pos_batch_after_mapping})')
      return jax.lax.pdot(x, y, axis_name=contract_names,
                          pos_batch=spec.pos_batch_after_mapping,
                          pos_contract=spec.pos_contract_after_mapping)

    fun = xmap(pdot_fun, in_axes=[lhs_spec, rhs_spec],
                out_axes=[*batch_names, ...], axis_resources=axis_resources)

    with generate_mesh(mesh_data):
      result = fun(lhs, rhs)

    expected = lax.dot_general(lhs, rhs, [spec.all_contract, spec.all_batch])
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
