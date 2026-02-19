import numpy as np


def tile_texture(texture, tile_size):
    """
    Function to tile a texture pattern using np.tile().
    :param texture: 2D array representing the texture pattern.
    :param tile_size: Tuple indicating the height and width of each tile.
    :return: 2D array with tiled texture applied.
    """
    tiled_texture = np.tile(texture, (tile_size[0], tile_size[1]))
    return tiled_texture


def apply_tiled_texture_to_floor(floor_area, texture, tile_size):
    """
    Applies the tiled texture to the floor area using a mask blending technique.
    :param floor_area: 2D array representing the floor area.
    :param texture: 2D array representing the texture pattern.
    :param tile_size: Tuple indicating the height and width of each tile.
    :return: 2D array of the floor area with the tiled texture applied.
    """
    tiled_texture = tile_texture(texture, tile_size)
    blended_floor = np.where(floor_area > 0, tiled_texture, floor_area)
    return blended_floor


def apply_perspective_tiled_texture(floor_area, texture, tile_size, perspective):
    """
    Applies perspective-correct tiled texture to the floor area.
    :param floor_area: 2D array representing the floor area.
    :param texture: 2D array representing the texture pattern.
    :param tile_size: Tuple indicating the height and width of each tile.
    :param perspective: Perspective transformation parameters.
    :return: Perspective transformed floor area with tiled texture applied.
    """
    # Placeholder: Add perspective transformation logic here.
    tiled_texture = tile_texture(texture, tile_size)
    # Example of applying perspective;
    # Apply transformation using perspective variables, then blend:
    blended_floor = np.where(floor_area > 0, tiled_texture, floor_area)
    return blended_floor


def apply_floor_texture_simple(floor_area, texture):
    """
    A simple helper function to directly apply a texture to the floor area.
    :param floor_area: 2D array representing the floor area.
    :param texture: 2D array representing the texture pattern.
    :return: 2D array of the floor area with the texture applied.
    """
    blended_floor = np.where(floor_area > 0, texture, floor_area)
    return blended_floor