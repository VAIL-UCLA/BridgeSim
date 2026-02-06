from enum import IntEnum


class SceneFrameType(IntEnum):
    """Intenum for scene frame types."""

    ORIGINAL = 0
    SYNTHETIC = 1


class StateSE2Index(IntEnum):
    """Intenum for SE(2) arrays."""

    X = 0
    Y = 1
    HEADING = 2

    @classmethod
    def size(cls):
        return 3

    @classmethod
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls.X, cls.Y + 1)

    @classmethod
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls.X, cls.HEADING + 1)


class BoundingBoxIndex(IntEnum):
    """Intenum of bounding boxes in logs."""

    X = 0
    Y = 1
    Z = 2
    LENGTH = 3
    WIDTH = 4
    HEIGHT = 5
    HEADING = 6

    @classmethod
    def size(cls):
        return 7

    @classmethod
    def POINT2D(cls):
        # assumes X, Y have subsequent indices
        return slice(cls.X, cls.Y + 1)

    @classmethod
    def POSITION(cls):
        # assumes X, Y, Z have subsequent indices
        return slice(cls.X, cls.Z + 1)

    @classmethod
    def DIMENSION(cls):
        # assumes LENGTH, WIDTH, HEIGHT have subsequent indices
        return slice(cls.LENGTH, cls.HEIGHT + 1)


class LidarIndex(IntEnum):
    """Intenum for lidar point cloud arrays."""

    _X = 0
    _Y = 1
    _Z = 2
    _INTENSITY = 3
    _RING = 4
    _ID = 5

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def Z(cls):
        return cls._Z

    @classmethod
    @property
    def INTENSITY(cls):
        return cls._INTENSITY

    @classmethod
    @property
    def RING(cls):
        return cls._RING

    @classmethod
    @property
    def ID(cls):
        return cls._ID

    @classmethod
    @property
    def POINT2D(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def POSITION(cls):
        # assumes X, Y, Z have subsequent indices
        return slice(cls._X, cls._Z + 1)