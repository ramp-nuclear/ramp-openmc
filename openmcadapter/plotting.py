from copy import deepcopy
from pathlib import Path, PurePath
from tempfile import TemporaryDirectory
from typing import Literal

import matplotlib as mpl
import numpy as np
import openmc
from coremaker.core import Core
from coremaker.geometries import Box
from coremaker.grid import cm
from coremaker.materials import Mixture
from coremaker.materials.water import make_light_water
from coremaker.tree import Tree, Node
from cytoolz import first
from matplotlib import pyplot as plt, image as mpimg
from matplotlib.figure import figaspect
from coremaker.protocols.geometry import Geometry
from matplotlib.pyplot import figure

from openmcadapter import openmc_core_to_model
from openmcadapter.geometry_adapter.element_adapter import \
    openmc_universe_from_element

COLOR_SPECIFIER = np.ndarray | str
MIXTURE_MAPPING = dict[Mixture, COLOR_SPECIFIER]
PATH_MAPPING = dict[PurePath, COLOR_SPECIFIER]
BASIS = Literal['xy', 'xz', 'yz']
STRUCTURE = Core | Tree | Geometry
DEFAULT_PIXELS_NUM = 400
DEFAULT_COLOR = 'blue'
# relative to figure size.
AXES_LIMIT = (0.15, 0.09, 0.775, 0.775)


def _get_limits(origin: tuple[cm, cm, cm],
                width: tuple[cm, cm],
                basis: BASIS) -> tuple[tuple[cm, cm], tuple[cm, cm]]:
    match basis:
        case 'xy':
            x, y = 0, 1
        case 'yz':
            # The x-axis will correspond to physical y and the y-axis will
            # correspond to physical z
            x, y = 1, 2
        case 'xz':
            # The y-axis will correspond to physical z
            x, y = 0, 2
    x_min = origin[x] - 0.5 * width[0]
    x_max = origin[x] + 0.5 * width[0]
    y_min = origin[y] - 0.5 * width[1]
    y_max = origin[y] + 0.5 * width[1]
    # noinspection PyUnboundLocalVariable
    return (x_min, x_max), (y_min, y_max)


def appropriate_resolution(pixels_num: int, picture_size: cm) -> cm:
    def f(i):
        return picture_size / i

    return (f(pixels_num) + f(pixels_num - 1)) / 2


def appropriate_pixels(resolution: cm, picture_size: cm) -> cm:
    return int(picture_size / resolution) + 1


def _plot_model(model: openmc.Model,
                width: tuple[cm, cm] = (4.0, 4.0),
                resolution: cm = None,
                origin: tuple[cm, cm, cm] = (0., 0., 0.),
                path: Path = None, basis: BASIS = 'xy',
                colors: dict[openmc.Cell, COLOR_SPECIFIER] = None,
                complement_color: COLOR_SPECIFIER = None, aspect='equal',
                ax: plt.Axes = None,
                **kwargs) -> \
        mpl.image.AxesImage | None:
    resolution = resolution or min(appropriate_resolution(DEFAULT_PIXELS_NUM,
                                                          size)
                                   for size in width)
    plot = openmc.Plot()
    plot.width = width
    plot.pixels = tuple(appropriate_pixels(resolution, size)
                        for size in width)
    plot.origin = origin
    plot.basis = basis
    if colors:
        if complement_color:
            cells = model.geometry.get_all_cells().values()
            for cell in filter(lambda x: x not in colors.keys(),
                               cells):
                colors[cell] = complement_color
        plot.colors = colors
        plot.color_by = 'cell'
    model.plots.append(plot)
    with TemporaryDirectory(prefix='openmc_plotter') as tempdir:
        if path:
            plot.filename = str(path.absolute())
            model.plot_geometry(output=False, cwd=tempdir)
            # renaming the output file from the openmc run to the supplied
            # path.
            path.absolute().with_name(f'{path.name}.png').rename(path)
        else:
            plot.filename = str(Path(f'{tempdir}/plot').absolute())
            model.plot_geometry(output=False, cwd=tempdir)
            img = mpimg.imread(f'{plot.filename}.png')
            if ax is None:
                fig_width, fig_height = figaspect(img)
                fig = figure(figsize=(fig_width, fig_height))
                fig: plt.Figure
                ax = fig.add_axes([.1, .1, 0.9, 0.9])
                params = fig.subplotpars
                dpi = plot.pixels[0] / (fig_width * (params.right - params.left))
                fig.set_dpi(dpi)
            (x_min, x_max), (y_min, y_max) = _get_limits(origin,
                                                         width,
                                                         basis)
            return ax.imshow(img, extent=(x_min, x_max, y_min, y_max),
                             interpolation='none', aspect=aspect, **kwargs)


def _bounding_geometry(structure: STRUCTURE) -> Geometry:
    match structure:
        case Core():
            geo = structure.outer_geometry
        case Tree():
            root, _ = first(structure.roots())
            geo = structure.geometry_of(root)
        case Geometry():
            geo = structure
    # noinspection PyUnboundLocalVariable
    return geo


def default_origin(structure: STRUCTURE) -> tuple[cm, cm, cm]:
    geo = _bounding_geometry(structure)
    try:
        bounding_box: Box = geo.bounding_box()
    except AttributeError:
        return (0.,) * 3

    return tuple(bounding_box.center)


def default_width(structure: STRUCTURE, basis: BASIS = 'xy') -> tuple[cm, cm]:
    geo = _bounding_geometry(structure)
    try:
        bounding_box: Box = geo.bounding_box()
    except AttributeError:
        # some random default width
        return (4.,) * 2
    match basis:
        case 'xy':
            x, y = 0, 1
        case 'yz':
            # The x-axis will correspond to physical y and the y-axis will
            # correspond to physical z
            x, y = 1, 2
        case 'xz':
            # The y-axis will correspond to physical z
            x, y = 0, 2
    # noinspection PyUnboundLocalVariable
    return bounding_box.dimensions[x], bounding_box.dimensions[y]


def plot_core(core: Core, width: tuple[cm, cm] = None,
              resolution: cm = None,
              origin: tuple[cm, cm, cm] = None,
              path: Path = None, basis: BASIS = 'xy',
              colors: MIXTURE_MAPPING | PATH_MAPPING = None,
              grid_color: COLOR_SPECIFIER = None, aspect='equal',
              ax: plt.Axes = None, sites_map=False, **kwargs) \
        -> mpl.image.AxesImage | None:
    f"""
    Plot a Core object.
    :param core: The Core object that models a reactor's core.
    :param width: The width of the region plotted in the directions defined
        by the chosen basis.
    :param resolution: The resolution (in cm) of the plot. By default,
        the resolution is chosen such that there will be at most 
        {DEFAULT_PIXELS_NUM} pixels in every plotting direction.
    :param colors: A mapping between component paths or mixtures and color
        specifiers (see openmc.plots for available options) to specify the color 
        of each component in the core model. There is no mixing, either the 
        colors are specified by mixtures or by node paths, and if so, 
        all node paths (or mixtures) present in the core must be supplied.
    :param origin: The location (given in the coordinates of the core object)
        around which a region is plotted.
    :param basis: A string to identify the cross section to be plotted.
        Can be either one of "xy", "yz" or "xz".
    :param path: The path (in the file system) to the png file of the plot.
        If no path is supplied, the image is not saved in a file and instead a 
        matplotlib Figure with the image is generated instead.
    :param grid_color: A color specifier to specify the assigned
        color to sites in the core's grid that are left empty. No input will
        result in random coloring. 
    :param aspect : {'equal', 'auto'} or float, default: :rc:`image.aspect`
       The aspect ratio of the Axes.  This parameter is particularly
       relevant for images since it determines whether data pixels are
       square.

       This parameter is a shortcut for explicitly calling
       `.Axes.set_aspect`. See there for further details.

       - 'equal': Ensures an aspect ratio of 1. Pixels will be square
          (unless pixel sizes are explicitly made non-square in data
          coordinates using *extent*).
       - 'auto': The Axes is kept fixed and the aspect is adjusted so
           that the data fit in the Axes. In general, this will result in
           non-square pixels. 
    :param ax : plt.Axes or None.
        An axes object on which to draw the plot. Some automatic features will
        not work if this is supplied 
        (like resizing the axes according to the plot's ratio).
    :param sites_map: bool
        Control the option to add the names of the sites of the grid at their places. Default is False.
    :param kwargs: passed to plt.imshow 
    :return: None if path is supplied or the mpl.image.AxesImage object
        of the requested image.
    """
    origin = origin if origin is not None else default_origin(core)
    width = width if width is not None else default_width(core, basis)
    model, cell_ids, _ = openmc_core_to_model(deepcopy(core), 'vacuum')
    cells = dict(model.geometry.get_all_cells())
    if colors:
        if all(isinstance(key, Mixture) for key in colors.keys()):
            colors = {path: colors[component.mixture]
                      for path, component in core.named_components}
        colors = {cells[cell_ids[path]]: colors[path]
                  for (path, node) in core.named_components}
    # It is assumed here that the only cells in the openmc model of the core
    # that are not in the cells dict are empty grid slots.
    ax = _plot_model(model, width, resolution, origin, path, basis,
                     colors, grid_color, aspect=aspect, ax=ax)
    if sites_map:
        for site in core.grid.sites():
            x, y = core.site_transform(site).translation[:2].flatten()
            ax.axes.text(x, y, site, ha='center', va='center')
    return ax


def _model_from_universe(universe: openmc.Universe) -> openmc.Model:
    geo = openmc.Geometry(root=universe)
    return openmc.Model(geometry=geo)


def plot_element(element: Tree,
                 width: tuple[cm, cm] = None,
                 resolution: cm = None,
                 origin: tuple[cm, cm, cm] = None,
                 path: Path = None, basis: BASIS = 'xy',
                 colors: MIXTURE_MAPPING | PATH_MAPPING = None, aspect='equal',
                 ax: plt.Axes = None) \
        -> mpl.image.AxesImage | None:
    f"""
    Plot a tree object that models a core element.
    :param element: The tree object that represents a core element.
    :param width: The width of the region plotted in the directions defined
        by the chosen basis.
    :param resolution: The resolution (in cm) of the plot. By default,
        the resolution is chosen such that there will be at most 
        {DEFAULT_PIXELS_NUM} pixels in every plotting direction.
    :param colors: A mapping between component paths or mixtures and color
        specifiers (see openmc.plots for available options) to specify the color 
        of each component in the core model. There is no mixing, either the 
        colors are specified by mixtures or by node paths, and if so, 
        all node paths (or mixtures) present in the core must be supplied.
    :param origin: The location (given in the coordinates of the core object)
        around which a region is plotted.
    :param basis: A string to identify the cross section to be plotted.
        Can be either one of "xy", "yz" or "xz".
    :param path: The path (in the file system) to the png file of the plot.
        Note that the file will have an extra ".png" suffix. If no path 
        is supplied, the image is not saved in a file and instead a matplotlib
        Figure with the image is generated instead.
    aspect : {'equal', 'auto'} or float, default: :rc:`image.aspect`
            The aspect ratio of the Axes.  This parameter is particularly
            relevant for images since it determines whether data pixels are
            square.

            This parameter is a shortcut for explicitly calling
            `.Axes.set_aspect`. See there for further details.

            - 'equal': Ensures an aspect ratio of 1. Pixels will be square
              (unless pixel sizes are explicitly made non-square in data
              coordinates using *extent*).
            - 'auto': The Axes is kept fixed and the aspect is adjusted so
              that the data fit in the Axes. In general, this will result in
              non-square pixels.
    :return: None if path is supplied or the mpl.image.AxesImage object
        of the requested image.
    """
    origin = origin if origin is not None else default_origin(element)
    width = width if width is not None else default_width(element, basis)
    element_name = 'element'
    element_path = PurePath(element_name)
    universe, cell_ids = openmc_universe_from_element(deepcopy(element),
                                                      element_name)
    if colors:
        if all(isinstance(key, Mixture) for key in colors.keys()):
            colors = {path: colors[component.mixture]
                      for path, component in element.named_components()}
        colors = {universe.cells[cell_ids[element_path / path]]: colors[path]
                  for (path, node) in element.named_components() if
                  element_path / path in cell_ids}
    model = _model_from_universe(universe)
    return _plot_model(model, width=width, resolution=resolution,
                       origin=origin, path=path, basis=basis,
                       colors=colors, aspect=aspect, ax=ax)


def plot_geometry(geometry: Geometry,
                  color: COLOR_SPECIFIER = None,
                  width: tuple[cm, cm] = None,
                  resolution: cm = None,
                  origin: tuple[cm, cm, cm] = None,
                  path: Path = None, basis: BASIS = 'xy', aspect='equal') \
        -> mpl.image.AxesImage | None:
    f"""
    Plot a Geometry object that models a region.
    :param geometry: The Geometry object that represents a region.
    :param width: The width of the region plotted in the directions defined
        by the chosen basis.
    :param resolution: The resolution (in cm) of the plot. By default,
        the resolution is chosen such that there will be at most 
        {DEFAULT_PIXELS_NUM} pixels in every plotting direction.
    :param color: The color of the plot.
    :param origin: The location (given in the coordinates of the core object)
        around which a region is plotted.
    :param basis: A string to identify the cross section to be plotted.
        Can be either one of "xy", "yz" or "xz".
    :param path: The path (in the file system) to the png file of the plot.
        Note that the file will have an extra ".png" suffix. If no path 
        is supplied, the image is not saved in a file and instead a matplotlib
        Figure with the image is generated instead.
    aspect : {'equal', 'auto'} or float, default: :rc:`image.aspect`
            The aspect ratio of the Axes.  This parameter is particularly
            relevant for images since it determines whether data pixels are
            square.

            This parameter is a shortcut for explicitly calling
            `.Axes.set_aspect`. See there for further details.

            - 'equal': Ensures an aspect ratio of 1. Pixels will be square
              (unless pixel sizes are explicitly made non-square in data
              coordinates using *extent*).
            - 'auto': The Axes is kept fixed and the aspect is adjusted so
              that the data fit in the Axes. In general, this will result in
              non-square pixels.
    :return: None if path is supplied or the mpl.image.AxesImage object
        of the requested image.
    """
    tree = Tree()
    color = color or DEFAULT_COLOR
    # node chosen randomly.
    node_path = PurePath('root_node')
    # The mixture is randomly chosen.
    mixture = make_light_water(20.)
    tree.nodes[node_path] = Node(geometry, mixture=mixture)
    return plot_element(tree, colors={mixture: color},
                        width=width, resolution=resolution,
                        origin=origin, path=path, basis=basis, aspect=aspect)
