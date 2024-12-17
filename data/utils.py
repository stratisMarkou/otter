ALL_GRID_SURFACE_DYNAMIC_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
]

ALL_GRID_MULTILEVEL_VARIABLES = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

ALL_GRID_STATIC_VARIABLES = [
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "land_sea_mask",
    "slope_of_sub_gridscale_orography",
    "geopotential_at_surface",
]

ALL_GRID_VARIABLE_NAMES = (
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES
    + ALL_GRID_MULTILEVEL_VARIABLES 
    + ALL_GRID_STATIC_VARIABLES
)

ALL_GRID_VARIABLE_LEVELS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]
