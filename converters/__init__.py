"""
Converters for transforming various autonomous driving datasets to ScenarioNet format.

Submodules:
- bench2drive: Bench2Drive (CARLA) -> ScenarioNet converter
- openscene: OpenScene (NuPlan) -> ScenarioNet converter
- waymo: Waymo Motion Dataset -> ScenarioNet converter
- nuscenes: nuScenes -> ScenarioNet converter
- common: Common utilities for dataset converters
"""

__all__ = ["bench2drive", "openscene", "waymo", "nuscenes", "common"]
