"""
Example OPC-UA client that reads sensor values and anomaly ground truth
"""
import asyncio
import time
from datetime import datetime
from asyncua import Client


async def read_single_sensor():
    """Example 1: Read a single sensor with its ground truth"""
    print("=" * 70)
    print("Example 1: Reading single sensor with ground truth")
    print("=" * 70)
    
    client = Client("opc.tcp://localhost:4840/freeopcua/server/")
    async with client:
        root = client.nodes.objects
        
        # Read LaserChamberPressure
        sensor = await root.get_child([
            "2:LaserDPM_01",
            "2:LaserChamberPressure"
        ])
        
        anomaly = await root.get_child([
            "2:LaserDPM_01",
            "2:LaserChamberPressure_Anomaly"
        ])
        
        # Monitor for 30 seconds
        print("\nMonitoring LaserChamberPressure for 30 seconds...")
        print(f"{'Timestamp':<20} {'Value':<12} {'Anomaly':<10}")
        print("-" * 50)
        
        for _ in range(300):  # 30 seconds at 10 Hz
            value = await sensor.read_value()
            is_anomaly = await anomaly.read_value()
            
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            anomaly_str = "⚠ TRUE" if is_anomaly else "FALSE"
            
            print(f"{timestamp:<20} {value:<12.3f} {anomaly_str:<10}")
            
            await asyncio.sleep(0.1)


async def read_all_sensors():
    """Example 2: Read all sensors with ground truth"""
    print("\n" + "=" * 70)
    print("Example 2: Reading all sensors with ground truth")
    print("=" * 70)
    
    client = Client("opc.tcp://localhost:4840/freeopcua/server/")
    async with client:
        root = client.nodes.objects
        
        # Get all machines
        machines = await root.get_children()
        
        print("\nAvailable machines and sensors:")
        for machine in machines:
            machine_name = (await machine.read_browse_name()).Name
            print(f"\n{machine_name}:")
            
            children = await machine.get_children()
            sensor_names = []
            for child in children:
                child_name = (await child.read_browse_name()).Name
                if not child_name.endswith("_Anomaly") and child_name != "Description":
                    sensor_names.append(child_name)
            
            for sensor_name in sensor_names:
                print(f"  - {sensor_name}")
        
        print("\n" + "-" * 70)
        print("Reading all sensors (5 iterations)...")
        print("-" * 70)
        
        for iteration in range(5):
            print(f"\nIteration {iteration + 1}:")
            
            for machine in machines:
                machine_name = (await machine.read_browse_name()).Name
                children = await machine.get_children()
                
                for child in children:
                    child_name = (await child.read_browse_name()).Name
                    
                    # Skip anomaly nodes and properties
                    if child_name.endswith("_Anomaly") or child_name == "Description":
                        continue
                    
                    # Read sensor value
                    value = await child.read_value()
                    
                    # Read anomaly ground truth
                    try:
                        anomaly_node = await machine.get_child([f"2:{child_name}_Anomaly"])
                        is_anomaly = await anomaly_node.read_value()
                        anomaly_flag = "🚨 ANOMALY" if is_anomaly else "✓ Normal"
                    except:
                        anomaly_flag = "N/A"
                    
                    print(f"  {machine_name}/{child_name}: {value:.3f} - {anomaly_flag}")
            
            await asyncio.sleep(2)


async def collect_dataset():
    """Example 3: Collect labeled dataset for training"""
    print("\n" + "=" * 70)
    print("Example 3: Collecting labeled dataset")
    print("=" * 70)
    
    client = Client("opc.tcp://localhost:4840/freeopcua/server/")
    async with client:
        root = client.nodes.objects
        
        dataset = []
        
        print("\nCollecting 100 samples...")
        
        for sample_num in range(100):
            machines = await root.get_children()
            
            for machine in machines:
                machine_name = (await machine.read_browse_name()).Name
                children = await machine.get_children()
                
                for child in children:
                    child_name = (await child.read_browse_name()).Name
                    
                    if child_name.endswith("_Anomaly") or child_name == "Description":
                        continue
                    
                    value = await child.read_value()
                    
                    try:
                        anomaly_node = await machine.get_child([f"2:{child_name}_Anomaly"])
                        is_anomaly = await anomaly_node.read_value()
                    except:
                        is_anomaly = False
                    
                    dataset.append({
                        "timestamp": time.time(),
                        "machine": machine_name,
                        "sensor": child_name,
                        "value": value,
                        "is_anomaly": is_anomaly
                    })
            
            if (sample_num + 1) % 10 == 0:
                print(f"  Collected {sample_num + 1} samples...")
            
            await asyncio.sleep(0.1)
        
        print(f"\nDataset collected: {len(dataset)} records")
        
        # Summary statistics
        anomaly_count = sum(1 for d in dataset if d["is_anomaly"])
        normal_count = len(dataset) - anomaly_count
        
        print(f"\nDataset summary:")
        print(f"  Total records: {len(dataset)}")
        print(f"  Normal samples: {normal_count} ({normal_count/len(dataset)*100:.1f}%)")
        print(f"  Anomaly samples: {anomaly_count} ({anomaly_count/len(dataset)*100:.1f}%)")
        
        # Show sample records
        print(f"\nSample records:")
        for i, record in enumerate(dataset[:5]):
            print(f"  {i+1}. {record['machine']}/{record['sensor']}: "
                  f"{record['value']:.3f} - Anomaly: {record['is_anomaly']}")
        
        return dataset


async def monitor_with_alerts():
    """Example 4: Monitor sensors and alert on anomalies"""
    print("\n" + "=" * 70)
    print("Example 4: Real-time monitoring with anomaly alerts")
    print("=" * 70)
    
    client = Client("opc.tcp://localhost:4840/freeopcua/server/")
    async with client:
        root = client.nodes.objects
        
        # Track anomaly states
        anomaly_states = {}
        
        print("\nMonitoring all sensors... (Press Ctrl+C to stop)")
        print("Alerts will be shown when anomalies start/end\n")
        
        try:
            while True:
                machines = await root.get_children()
                
                for machine in machines:
                    machine_name = (await machine.read_browse_name()).Name
                    children = await machine.get_children()
                    
                    for child in children:
                        child_name = (await child.read_browse_name()).Name
                        
                        if child_name.endswith("_Anomaly") or child_name == "Description":
                            continue
                        
                        value = await child.read_value()
                        
                        try:
                            anomaly_node = await machine.get_child([f"2:{child_name}_Anomaly"])
                            is_anomaly = await anomaly_node.read_value()
                        except:
                            continue
                        
                        sensor_key = f"{machine_name}/{child_name}"
                        previous_state = anomaly_states.get(sensor_key, False)
                        
                        # Detect state changes
                        if is_anomaly and not previous_state:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"[{timestamp}] 🚨 ANOMALY STARTED: {sensor_key} (value: {value:.3f})")
                        
                        elif not is_anomaly and previous_state:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"[{timestamp}] ✓ ANOMALY ENDED: {sensor_key} (value: {value:.3f})")
                        
                        anomaly_states[sensor_key] = is_anomaly
                
                await asyncio.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")


async def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("OPC-UA Client Examples - Sensor Data with Anomaly Ground Truth")
    print("=" * 70)
    print("\nMake sure the OPC-UA server is running before starting!\n")
    
    try:
        # Example 1: Read single sensor
        await read_single_sensor()
        
        # Example 2: Read all sensors
        await read_all_sensors()
        
        # Example 3: Collect dataset
        dataset = await collect_dataset()
        
        # Example 4: Monitor with alerts
        await monitor_with_alerts()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the OPC-UA server is running:")
        print("  python opcua_sensor_server.py")


if __name__ == "__main__":
    asyncio.run(main())