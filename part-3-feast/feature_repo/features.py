# Dynamically generated Feast Feature Definitions
from datetime import timedelta
from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.value_type import ValueType
from feast.types import Float64

# 1. Define Entity
sample_entity = Entity(
    name="sample_id",
    join_keys=["sample_id"],
    value_type=ValueType.INT64,
    description="Unique identifier for each sample instance",
)

# 2. Define Batch Source
features_source = FileSource(
    name="features_source",
    path="data/features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 3. Define Raw Features FeatureView
raw_features_fv = FeatureView(
    name="raw_features_fv",
    entities=[sample_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="pixel_0_0", dtype=Float64),
        Field(name="pixel_0_1", dtype=Float64),
        Field(name="pixel_0_2", dtype=Float64),
        Field(name="pixel_0_3", dtype=Float64),
        Field(name="pixel_0_4", dtype=Float64),
        Field(name="pixel_0_5", dtype=Float64),
        Field(name="pixel_0_6", dtype=Float64),
        Field(name="pixel_0_7", dtype=Float64),
        Field(name="pixel_1_0", dtype=Float64),
        Field(name="pixel_1_1", dtype=Float64),
        Field(name="pixel_1_2", dtype=Float64),
        Field(name="pixel_1_3", dtype=Float64),
        Field(name="pixel_1_4", dtype=Float64),
        Field(name="pixel_1_5", dtype=Float64),
        Field(name="pixel_1_6", dtype=Float64),
        Field(name="pixel_1_7", dtype=Float64),
        Field(name="pixel_2_0", dtype=Float64),
        Field(name="pixel_2_1", dtype=Float64),
        Field(name="pixel_2_2", dtype=Float64),
        Field(name="pixel_2_3", dtype=Float64),
        Field(name="pixel_2_4", dtype=Float64),
        Field(name="pixel_2_5", dtype=Float64),
        Field(name="pixel_2_6", dtype=Float64),
        Field(name="pixel_2_7", dtype=Float64),
        Field(name="pixel_3_0", dtype=Float64),
        Field(name="pixel_3_1", dtype=Float64),
        Field(name="pixel_3_2", dtype=Float64),
        Field(name="pixel_3_3", dtype=Float64),
        Field(name="pixel_3_4", dtype=Float64),
        Field(name="pixel_3_5", dtype=Float64),
        Field(name="pixel_3_6", dtype=Float64),
        Field(name="pixel_3_7", dtype=Float64),
        Field(name="pixel_4_0", dtype=Float64),
        Field(name="pixel_4_1", dtype=Float64),
        Field(name="pixel_4_2", dtype=Float64),
        Field(name="pixel_4_3", dtype=Float64),
        Field(name="pixel_4_4", dtype=Float64),
        Field(name="pixel_4_5", dtype=Float64),
        Field(name="pixel_4_6", dtype=Float64),
        Field(name="pixel_4_7", dtype=Float64),
        Field(name="pixel_5_0", dtype=Float64),
        Field(name="pixel_5_1", dtype=Float64),
        Field(name="pixel_5_2", dtype=Float64),
        Field(name="pixel_5_3", dtype=Float64),
        Field(name="pixel_5_4", dtype=Float64),
        Field(name="pixel_5_5", dtype=Float64),
        Field(name="pixel_5_6", dtype=Float64),
        Field(name="pixel_5_7", dtype=Float64),
        Field(name="pixel_6_0", dtype=Float64),
        Field(name="pixel_6_1", dtype=Float64),
        Field(name="pixel_6_2", dtype=Float64),
        Field(name="pixel_6_3", dtype=Float64),
        Field(name="pixel_6_4", dtype=Float64),
        Field(name="pixel_6_5", dtype=Float64),
        Field(name="pixel_6_6", dtype=Float64),
        Field(name="pixel_6_7", dtype=Float64),
        Field(name="pixel_7_0", dtype=Float64),
        Field(name="pixel_7_1", dtype=Float64),
        Field(name="pixel_7_2", dtype=Float64),
        Field(name="pixel_7_3", dtype=Float64),
        Field(name="pixel_7_4", dtype=Float64),
        Field(name="pixel_7_5", dtype=Float64),
        Field(name="pixel_7_6", dtype=Float64),
        Field(name="pixel_7_7", dtype=Float64),
    ],
    online=True,
    source=features_source,
    tags={"tier": "raw", "dataset": "digits"},
)

# 4. Define Engineered Features FeatureView
engineered_features_fv = FeatureView(
    name="engineered_features_fv",
    entities=[sample_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="feat_interaction_prod", dtype=Float64),
        Field(name="feat_interaction_ratio", dtype=Float64),
    ],
    online=True,
    source=features_source,
    tags={"tier": "engineered"},
)

# 5. Define FeatureService for Cross-Model Reusability
exam_feature_service = FeatureService(
    name="exam_feature_service",
    features=[raw_features_fv, engineered_features_fv],
    tags={"team": "mlops-exam"},
)
