from common.registry import Registry
ScenarioClassRegistry = Registry(name='ScenarioClassRegistry')

from .change_lane import ChangeLane
from .control_loss import ControlLoss
from .follow_leading_vehicle import FollowLeadingVehicle, FollowLeadingVehicleWithObstacle
from .junction_crossing_route import SignalJunctionCrossingRoute, NoSignalJunctionCrossingRoute
from .maneuver_opposite_direction import ManeuverOppositeDirection
from .no_signal_junction_crossing import NoSignalJunctionCrossing
from .object_crash_intersection import VehicleTurningRight, VehicleTurningLeft, VehicleTurningRoute
from .object_crash_vehicle import DynamicObjectCrossing, StationaryObjectCrossing
from .opposite_vehicle_taking_priority import OppositeVehicleRunningRedLight
from .other_leading_vehicle import OtherLeadingVehicle
from .signalized_junction_left_turn import SignalizedJunctionLeftTurn
from .signalized_junction_right_turn import SignalizedJunctionRightTurn
from .construction_crash_vehicle import ConstructionSetupCrossing
from .change_lane import ChangeLane
from .cut_in import CutIn
from .freeride import FreeRide