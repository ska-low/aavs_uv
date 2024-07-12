"""aa_module: A basic class for child objects."""

class AaBaseModule(object):
    """A basic class for ApertureArray child objects.

    Anything in a child Classes docstring will be added to the __repr__.
    Classes inheriting from this should have a name starting with `Aa`,
    e.g. `AaCoords`. They are 'attached' to the ApertureArray thusly::

        Class ApertureArray(object):
        def __init__(self, ...):
            self.coords       = AaCoords(self)

    When calling display() in ipython / jupyter, the __repr__ will print
    all the stuff in the docstring -- which should be useful to the user.
    e.g.::

        <Aperture Array module: coords>
        Coordinate utils.

        Provides the following:
            get_sun() - Get the position of the sun as a SkyCoord
            get_zenith() - Get the zenith as a SkyCoord
            get_alt_az() - Get the alt/az of a given SkyCoord
            generate_phase_vector() - Generate a phase vector toward a given SkyCoord
    """
    def __init__(self, name):
        """Initialize class.

        Args:
            name (str): Name of module.
        """
        self.name = name

    def __repr__(self):
        """Print simple representation of class."""
        return f"<Aperture Array module: {self.name}>\n" + self.__doc__
