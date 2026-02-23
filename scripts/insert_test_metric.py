from datetime import datetime
from src.db.session import SessionLocal
from src.db.models import ProcessMetric

def main():
    session = SessionLocal()

    metric = ProcessMetric(
        timestamp=datetime.utcnow(),
        station_name="shuttle_station",
        metric_name="shuttle_pre_scan_time",
        value=0.012,
        unit="s",
        state_context="PRE_SCANNING"
    )

    session.add(metric)
    session.commit()
    session.close()

    print("Inserted successfully.")

if __name__ == "__main__":
    main()

