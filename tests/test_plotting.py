import pytest
from coremaker.example import example_core, site_size, U_mixture, heavy_water, \
    aluminium, hafnium, lattice_shape, hafnium_block_size
from coremaker.geometries import Box, Annulus
from more_itertools import first
from openmcadapter.plotting import *


def test_plot_geometry(image_regression):
    outer_radius = 1.
    geometry = Annulus(center=(0., 0., 0.),
                       inner_radius=0.5, outer_radius=outer_radius,
                       length=20., axis=(0., 0., 1.))
    with TemporaryDirectory() as d:
        path = Path(d) / Path('plot')
        plot_geometry(geometry,
                      path=path,
                      width=(2 * outer_radius, ) * 2)
        image_regression.check(path.read_bytes())


@pytest.fixture()
def colors():
    colors = {heavy_water: 'blue',
              aluminium: 'gray',
              U_mixture: 'red',
              hafnium: 'black'}
    return colors


def test_plot_element(image_regression, colors):
    key = first(example_core.grid.keys())
    element = example_core.grid[key]
    assert set(colors.keys()) >= set(comp.mixture
                                     for path, comp in element.named_components())
    with TemporaryDirectory() as d:
        path = Path(d) / 'plot'
        plot_element(element,
                     path=path,
                     width=(site_size, ) * 2,
                     colors=colors)
        image_regression.check(path.read_bytes())


@pytest.mark.parametrize('basis', ['xy', 'yz', 'xz'])
def test_plot_core(image_regression, colors, basis):
    assert set(colors.keys()) >= set(comp.mixture
                                     for path, comp in example_core.named_components)
    with TemporaryDirectory() as d:
        path = Path(d) / 'plot'
        # noinspection PyTypeChecker
        plot_core(example_core,
                  path=path,
                  width=tuple(3 / 2 * shape * site_size + hafnium_block_size
                              for shape in lattice_shape),
                  colors=colors,
                  basis=basis)
        image_regression.check(path.read_bytes(),
                               basename=f'test_plot_core_{basis}')


def test_color_empty_grid_slots(image_regression, colors):
    core = deepcopy(example_core)
    for site in core.grid.sites():
        if 'A' in site:
            del core.grid[site]
    with TemporaryDirectory() as d:
        path = Path(d) / 'plot'
        # noinspection PyTypeChecker
        plot_core(core,
                  path=path,
                  width=tuple(shape * site_size for shape in lattice_shape),
                  colors=colors,
                  basis='xy',
                  grid_color='purple')
        image_regression.check(path.read_bytes())


def test_plotting_with_large_aspect(image_regression, colors):
    core = deepcopy(example_core)
    axes = plot_core(core,
                     width=(site_size * lattice_shape[0], site_size / 60),
                     resolution=site_size / 120,
                     colors=colors,
                     basis='xy')
    fig = axes.get_figure()
    with TemporaryDirectory() as d:
        path = Path(d) / 'figure.png'
        fig.savefig(path)
        image_regression.check(path.read_bytes())
