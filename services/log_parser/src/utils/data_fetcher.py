# src/utils/data_fetcher.py
import json
import time
import os
from abc import ABC, abstractmethod

class BaseConsumer(ABC):
    """
    The top-level contract. 
    It doesn't assume files, sockets, or anything else.
    """
    @abstractmethod
    def stream(self):
        """Yields clean strings to the parser."""
        pass

# =================================================================
# DISK-BASED LOGIC (Shared by Vector Buffer and Raw Files)
# =================================================================

class FileStreamer(BaseConsumer):
    """
    A specialized base for anything that lives on a disk.
    This is where the 'open()' and 'tail' logic lives.
    """
    def __init__(self, path, live=True):
        self.path = path
        self.live = live

    @abstractmethod
    def _normalize(self, line):
        """Subclasses handle the string vs JSON logic."""
        pass

    def stream(self):
        if not os.path.exists(self.path):
            return

        with open(self.path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    if self.live:
                        time.sleep(0.1)
                        continue
                    break
                
                clean_msg = self._normalize(line)
                if clean_msg:
                    yield clean_msg

# =================================================================
# CONCRETE IMPLEMENTATIONS
# =================================================================

class RawFileConsumer(FileStreamer):
    """How to handle raw local files."""
    def _normalize(self, line):
        return line.strip()

class VectorBufferConsumer(FileStreamer):
    """How to handle Vector NDJSON files."""
    def _normalize(self, line):
        try:
            return json.loads(line.strip()).get('message')
        except:
            return None

class KafkaConsumer(BaseConsumer):
    """
    NOTICE: This inherits from BaseConsumer, NOT FileStreamer.
    It doesn't use open() or f.readline().
    """
    def __init__(self, topic, servers):
        self.topic = topic
        self.servers = servers

    def stream(self):
        # Kafka-specific network logic (no file handling here!)
        from kafka import KafkaConsumer as _KC
        consumer = _KC(self.topic, bootstrap_servers=self.servers)
        for msg in consumer:
            # Unwraps its own JSON/Bytes
            yield msg.value.decode('utf-8')