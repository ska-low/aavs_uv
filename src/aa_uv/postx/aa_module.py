
class AaBaseModule(object):
    """A basic class for ApertureArray child objects"""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<Aperture Array module: {self.name}>\n" + self.__doc__
