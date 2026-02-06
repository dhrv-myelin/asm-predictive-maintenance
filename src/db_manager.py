"""
src/db_manager.py
Simple TimescaleDB Manager - Just connects and creates tables
"""
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import json
import os

class TimescaleDBManager:
    """Manages TimescaleDB connection and inserts"""
    
    def __init__(self, config_path="config/db_config.json"):
        self.config = self._load_config(config_path)
        self.engine = None
        self.enabled = False
        
        # Table names
        self.metrics_table = "process_metrics"
        self.errors_table = "error_logs"
        self.baseline_table = "baseline_metrics"  # Statistical baseline data from process_baseline.json
        
        print(f"[DB INIT] Config loaded - enabled={self.config.get('enabled', False)}")
        
        if self.config.get("enabled", False):
            self._connect_and_setup()
        else:
            print("[DB INIT] Database disabled in config")
        
        print(f"[DB INIT] Final status - self.enabled={self.enabled}")
    
    def _load_config(self, config_path):
        """Load database configuration from JSON"""
        if not os.path.exists(config_path):
            print(f"✗ Config not found: {config_path}")
            return {"enabled": False}
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"✓ Config loaded from {config_path}")
                print(f"  Host: {config.get('host')}")
                print(f"  Port: {config.get('port')}")
                print(f"  Database: {config.get('database')}")
                print(f"  User: {config.get('user')}")
                print(f"  Enabled: {config.get('enabled')}")
                return config
        except Exception as e:
            print(f"✗ Failed to load config: {e}")
            return {"enabled": False}
    
    def _connect_and_setup(self):
        """Connect to database and create tables"""
        print("\n=== Attempting Database Connection ===")
        
        try:
            # Build connection string
            conn_str = (
                f"postgresql+psycopg2://{self.config['user']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            
            print(f"Connecting to: {self.config['host']}:{self.config['port']}/{self.config['database']}")
            
            # Create engine
            self.engine = create_engine(
                conn_str,
                poolclass=QueuePool,
                pool_size=5,
                pool_pre_ping=True
            )
            
            print("Testing connection...")
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                print(f"✓ Connected! PostgreSQL version: {version[:50]}")
            
            print(f"✓ Connection successful to database: {self.config['database']}")
            
            # Create tables
            print("\nCreating tables...")
            self._create_tables()
            
            self.enabled = True
            print("✓ Database fully initialized and ready")
            print("=" * 50)
            
        except Exception as e:
            print(f"\n✗ DATABASE CONNECTION FAILED")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("\nCommon causes:")
            print("  1. PostgreSQL is not running")
            print("  2. Wrong password")
            print("  3. Database doesn't exist yet")
            print("  4. psycopg2 not installed: pip install psycopg2-binary")
            print("=" * 50)
            
            import traceback
            traceback.print_exc()
            
            self.enabled = False
            
            if self.engine:
                self.engine.dispose()
                self.engine = None
    
    def _create_tables(self):
        """Create tables if they don't exist"""
        try:
            # Use begin() for proper transaction handling
            with self.engine.begin() as conn:
                # 1. Real-time Metrics table
                print(f"  Creating {self.metrics_table}...")
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.metrics_table} (
                        timestamp TIMESTAMPTZ NOT NULL,
                        component_id TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value DOUBLE PRECISION,
                        unit TEXT,
                        state_context TEXT
                    )
                """))
                
                # 2. Errors table
                print(f"  Creating {self.errors_table}...")
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.errors_table} (
                        timestamp TIMESTAMPTZ NOT NULL,
                        station_name TEXT,
                        severity TEXT NOT NULL,
                        service_name TEXT,
                        log_message TEXT,
                        extra TEXT
                    )
                """))
                
                # 3. Baseline metrics table (statistical data from process_baseline.json)
                print(f"  Creating {self.baseline_table}...")
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.baseline_table} (
                        id SERIAL PRIMARY KEY,
                        station_name TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        mean DOUBLE PRECISION,
                        std_dev DOUBLE PRECISION,
                        min_limit DOUBLE PRECISION,
                        max_limit DOUBLE PRECISION,
                        sample_count INTEGER,
                        unit TEXT DEFAULT 'seconds',
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(station_name, metric_name)
                    )
                """))
                
                # Create indexes
                print(f"  Creating indexes...")
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_metrics_component 
                        ON {self.metrics_table} (component_id, timestamp DESC)
                """))
                
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_errors_severity 
                        ON {self.errors_table} (severity, timestamp DESC)
                """))
                
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_baseline_station 
                        ON {self.baseline_table} (station_name, metric_name)
                """))
                
                # Transaction is automatically committed when exiting 'with' block
                print(f"✓ Tables and indexes created successfully")
                
        except Exception as e:
            print(f"✗ Table creation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def batch_insert_metrics(self, metrics_batch):
        """Insert multiple metrics at once"""
        if not self.enabled:
            print("[DB] Skipping batch insert - database not enabled")
            return
            
        if not metrics_batch:
            return
        
        try:
            batch_dicts = []
            for row in metrics_batch:
                if row is None:
                    continue
                batch_dicts.append({
                    "ts": row[0],
                    "comp": row[1],
                    "metric": row[2],
                    "val": row[3],
                    "unit": row[4],
                    "ctx": row[5]
                })
            
            if not batch_dicts:
                return
            
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO {self.metrics_table} 
                    (timestamp, component_id, metric_name, value, unit, state_context)
                    VALUES (:ts, :comp, :metric, :val, :unit, :ctx)
                """), batch_dicts)
                conn.commit()
            
            print(f"[DB] ✓ Inserted {len(batch_dicts)} metrics")
                
        except Exception as e:
            print(f"[DB] ✗ Batch insert error: {e}")

    def insert_metric(self, timestamp, component_id, metric_name, value, unit, state_context):
        """Insert a single metric - STREAMING"""
        if not self.enabled:
            print(f"[DB] ✗ Cannot insert - database not enabled")
            return
        
        if not self.engine:
            print(f"[DB] ✗ Cannot insert - no engine connection")
            return
            
        try:
            print(f"[DB] Attempting insert: {component_id}.{metric_name} = {value}")
            
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    INSERT INTO {self.metrics_table} 
                    (timestamp, component_id, metric_name, value, unit, state_context)
                    VALUES (:ts, :comp, :metric, :val, :unit, :ctx)
                """), {
                    "ts": timestamp,
                    "comp": component_id,
                    "metric": metric_name,
                    "val": value,
                    "unit": unit,
                    "ctx": state_context
                })
                conn.commit()
                
                print(f"[DB] ✓ Insert successful")
                    
        except Exception as e:
            print(f"[DB] ✗ Insert error: {e}")
            import traceback
            traceback.print_exc()
    
    def insert_error(self, timestamp, station_name, severity, service_name, log_message, extra):
        """Insert error log"""
        if not self.enabled:
            return
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO {self.errors_table}
                    (timestamp, station_name, severity, service_name, log_message, extra)
                    VALUES (:ts, :station, :severity, :service, :message, :extra)
                """), {
                    "ts": timestamp,
                    "station": station_name,
                    "severity": severity,
                    "service": service_name,
                    "message": log_message,
                    "extra": extra
                })
                conn.commit()
                
        except Exception as e:
            print(f"[DB] ✗ Error insert failed: {e}")
    
    # ============================================================================
    # BASELINE METRICS METHODS (process_baseline.json)
    # ============================================================================
    
    def load_baseline_from_json(self, json_path="data/process_baseline.json"):
        """
        Load baseline metrics from process_baseline.json and insert into database.
        Uses UPSERT (INSERT ... ON CONFLICT) to update if record exists.
        
        Args:
            json_path: Path to the process_baseline.json file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            print("[DB] ✗ Database not enabled - cannot load baseline")
            return False
        
        if not os.path.exists(json_path):
            print(f"[DB] ⚠ Baseline file not found: {json_path}")
            return False
        
        try:
            print(f"\n[DB] Loading baseline metrics from {json_path}...")
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            total_inserted = 0
            
            # Iterate through each station in the JSON
            for station_name, metrics in data.items():
                for metric_name, stats in metrics.items():
                    self.upsert_baseline_metric(
                        station_name=station_name,
                        metric_name=metric_name,
                        mean=stats.get('mean'),
                        std_dev=stats.get('std_dev'),
                        min_limit=stats.get('min_limit'),
                        max_limit=stats.get('max_limit'),
                        sample_count=stats.get('sample_count'),
                        unit='seconds'
                    )
                    total_inserted += 1
            
            print(f"[DB] ✓ Loaded {total_inserted} baseline metrics from {json_path}")
            return True
            
        except Exception as e:
            print(f"[DB] ✗ Failed to load baseline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def upsert_baseline_metric(self, station_name, metric_name, mean, std_dev, 
                                min_limit, max_limit, sample_count, unit='seconds'):
        """
        Insert or update a single baseline metric.
        Uses PostgreSQL's ON CONFLICT clause for upsert functionality.
        
        Args:
            station_name: Equipment/station identifier
            metric_name: Specific timing metric name
            mean: Average value
            std_dev: Standard deviation
            min_limit: Minimum observed value
            max_limit: Maximum observed value
            sample_count: Number of samples collected
            unit: Unit of measurement (default: 'seconds')
        """
        if not self.enabled:
            return
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO {self.baseline_table}
                    (station_name, metric_name, mean, std_dev, min_limit, max_limit, sample_count, unit, updated_at)
                    VALUES (:station, :metric, :mean, :std_dev, :min_limit, :max_limit, :sample_count, :unit, CURRENT_TIMESTAMP)
                    ON CONFLICT (station_name, metric_name) 
                    DO UPDATE SET 
                        mean = EXCLUDED.mean,
                        std_dev = EXCLUDED.std_dev,
                        min_limit = EXCLUDED.min_limit,
                        max_limit = EXCLUDED.max_limit,
                        sample_count = EXCLUDED.sample_count,
                        unit = EXCLUDED.unit,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    "station": station_name,
                    "metric": metric_name,
                    "mean": mean,
                    "std_dev": std_dev,
                    "min_limit": min_limit,
                    "max_limit": max_limit,
                    "sample_count": sample_count,
                    "unit": unit
                })
                conn.commit()
                
        except Exception as e:
            print(f"[DB] ✗ Baseline upsert failed for {station_name}.{metric_name}: {e}")
    
    def get_baseline_metric(self, station_name, metric_name):
        """
        Retrieve a single baseline metric from the database.
        
        Args:
            station_name: Equipment/station identifier
            metric_name: Specific timing metric name
            
        Returns:
            dict: Baseline metrics or None if not found
        """
        if not self.enabled:
            print("[DB] ✗ Database not enabled")
            return None
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT mean, std_dev, min_limit, max_limit, sample_count, unit
                    FROM {self.baseline_table}
                    WHERE station_name = :station AND metric_name = :metric
                """), {
                    "station": station_name,
                    "metric": metric_name
                })
                row = result.fetchone()
                
                if row:
                    return {
                        'mean': row[0],
                        'std_dev': row[1],
                        'min_limit': row[2],
                        'max_limit': row[3],
                        'sample_count': row[4],
                        'unit': row[5]
                    }
                return None
                
        except Exception as e:
            print(f"[DB] ✗ Failed to get baseline: {e}")
            return None
    
    def get_all_baseline_metrics(self, station_name=None):
        """
        Retrieve all baseline metrics, optionally filtered by station.
        
        Args:
            station_name: Optional filter by station name
            
        Returns:
            list: List of baseline metric dictionaries
        """
        if not self.enabled:
            print("[DB] ✗ Database not enabled")
            return []
        
        try:
            with self.engine.connect() as conn:
                if station_name:
                    result = conn.execute(text(f"""
                        SELECT station_name, metric_name, mean, std_dev, min_limit, max_limit, sample_count, unit
                        FROM {self.baseline_table}
                        WHERE station_name = :station
                        ORDER BY station_name, metric_name
                    """), {"station": station_name})
                else:
                    result = conn.execute(text(f"""
                        SELECT station_name, metric_name, mean, std_dev, min_limit, max_limit, sample_count, unit
                        FROM {self.baseline_table}
                        ORDER BY station_name, metric_name
                    """))
                
                rows = result.fetchall()
                return [
                    {
                        'station_name': row[0],
                        'metric_name': row[1],
                        'mean': row[2],
                        'std_dev': row[3],
                        'min_limit': row[4],
                        'max_limit': row[5],
                        'sample_count': row[6],
                        'unit': row[7]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            print(f"[DB] ✗ Failed to get baselines: {e}")
            return []
    
    def export_baseline_to_json(self, output_path="data/process_baseline_export.json"):
        """
        Export baseline metrics from database back to JSON format.
        
        Args:
            output_path: Path to save the exported JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            print("[DB] ✗ Database not enabled")
            return False
        
        try:
            metrics = self.get_all_baseline_metrics()
            
            # Reorganize into nested structure matching input format
            output = {}
            for m in metrics:
                station = m['station_name']
                metric = m['metric_name']
                
                if station not in output:
                    output[station] = {}
                
                output[station][metric] = {
                    'mean': m['mean'],
                    'std_dev': m['std_dev'],
                    'min_limit': m['min_limit'],
                    'max_limit': m['max_limit'],
                    'sample_count': m['sample_count']
                }
            
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"[DB] ✓ Exported baseline metrics to {output_path}")
            return True
            
        except Exception as e:
            print(f"[DB] ✗ Export failed: {e}")
            return False
    
    # ============================================================================
    # EXISTING METHODS
    # ============================================================================
    
    def get_metrics_for_baseline(self):
        """Read all metrics from database"""
        print(f"[DB] get_metrics_for_baseline called - enabled={self.enabled}")
        
        if not self.enabled:
            print("[DB] ✗ Database not enabled - cannot read metrics")
            return []
        
        try:
            print("  Querying database for all metrics...")
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT component_id, metric_name, value
                    FROM {self.metrics_table}
                    ORDER BY timestamp
                """))
                rows = result.fetchall()
                print(f"[DB] ✓ Retrieved {len(rows)} rows from database")
                return rows
                
        except Exception as e:
            print(f"[DB] ✗ Read error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def close(self):
        """Close connection"""
        if self.engine:
            self.engine.dispose()
            print("[DB] ✓ Database closed")