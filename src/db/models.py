from sqlalchemy import (
    Column,
    Integer,
    Text,
    String,
    Float,
    TIMESTAMP,
    UniqueConstraint,
    Index
)
from sqlalchemy.sql import func
from src.db.base import Base
from sqlalchemy import BigInteger, Identity

# ===============================
# process_metrics
# ===============================

class ProcessMetric(Base):
    __tablename__ = "process_metrics"

    id = Column(BigInteger, Identity(always=True), primary_key=True)

    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    station_name = Column(Text, nullable=False)
    metric_name = Column(Text, nullable=False)
    value = Column(Float)
    unit = Column(Text)
    state_context = Column(Text)
    #new_col
    status = Column(Text, nullable=True) 

    __table_args__ = (
        Index(
            "idx_metrics_component",
            "station_name",
            "timestamp"
        ),
    )


# ===============================
# error_logs
# ===============================
class ErrorLog(Base):
    __tablename__ = "error_logs"

    id = Column(BigInteger, Identity(always=True), primary_key=True)

    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    station_name = Column(Text)
    severity = Column(Text, nullable=False)
    service_name = Column(Text)
    log_message = Column(Text)
    extra = Column(Text)

    __table_args__ = (
        Index(
            "idx_errors_severity",
            "severity",
            "timestamp"
        ),
    )



# ===============================
# baseline_metrics
# ===============================

class BaselineMetric(Base):
    __tablename__ = "baseline_metrics"

    id = Column(Integer, primary_key=True)

    station_name = Column(Text, nullable=False)
    metric_name = Column(Text, nullable=False)

    mean = Column(Float)
    std_dev = Column(Float)
    min_limit = Column(Float)
    max_limit = Column(Float)
    sample_count = Column(Integer)


    unit = Column(Text, default="seconds")

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.current_timestamp()
    )

    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )

    __table_args__ = (
        UniqueConstraint(
            "station_name",
            "metric_name",
            name="uq_baseline_station_metric"
        ),
        Index(
            "idx_baseline_station",
            "station_name",
            "metric_name"
        ),
    )
    
    
# ===============================
# model_predictions
# ===============================

class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id = Column(BigInteger, Identity(always=True), primary_key=True)

    actual_timestamp = Column(
        TIMESTAMP(timezone=True),
        nullable=False
    )

    predicted_timestamp = Column(
        TIMESTAMP(timezone=True),
        nullable=False
    )

    predicted_value = Column(
        Float,
        nullable=False
    )

    station_name = Column(Text, nullable=False)
    metric_name = Column(Text, nullable=False)
    model_name = Column(Text, nullable=False)

    __table_args__ = (
        Index(
            "idx_model_predictions_station_metric",
            "station_name",
            "metric_name",
            "predicted_timestamp"
        ),
    )
    
