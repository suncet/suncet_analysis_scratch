"""
Spacecraft Battery Simulation

This script simulates the battery state of charge for a tumbling spacecraft
that starts with a depleted battery and charges via solar panels while
periodically attempting deployments of antennas and solar panels.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PowerEvent:
    """Represents a power consumption event."""
    name: str
    power_consumption: float  # Watts
    duration: float  # seconds
    periodicity: int  # seconds


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    simulation_duration: int = 24 * 60 * 60  # seconds (24 hours)
    time_step: int = 5  # seconds
    battery_capacity: float = 83.52  # Watt-hours
    initial_state_of_charge: float = 0.0  # Watt-hours
    min_voltage: float = 12.0  # Volts
    max_voltage: float = 16.8  # Volts
    battery_voltages: np.ndarray = np.array([
        12.170, 12.364, 12.559, 12.754, 12.949, 13.144, 13.339, 13.534, 13.632,
        13.730, 13.828, 13.926, 14.024, 14.121, 14.219, 14.317, 14.415, 14.513,
        14.611, 14.709, 14.807, 14.906, 15.004, 15.102, 15.209, 15.320, 15.432,
        15.543, 15.654, 15.765, 15.877, 15.988, 16.099
    ])  # Battery voltages
    solar_panel_powers: np.ndarray = np.array([
        10.71, 11.11, 11.52, 11.92, 12.33, 12.74, 13.14, 13.55, 13.75,
        13.95, 14.15, 14.36, 14.56, 14.76, 14.97, 15.17, 15.37, 15.57,
        15.78, 15.98, 16.18, 16.39, 16.59, 16.79, 16.49, 15.98, 15.47,
        14.94, 14.40, 13.86, 13.30, 10.80, 5.43
    ])  # Solar panel power available to charge battery at each battery voltage (Watts)
    tumbling_efficiency_factor: float = 0.25  # Efficiency due to tumbling. It works out to this number whether the panels are deployed or not. This accounts for the cosine of the angle between the sun and the panel as well. 


class BatterySimulator:
    """Simulates spacecraft battery state of charge over time."""
    
    def __init__(self, config: SimulationConfig, power_events: List[PowerEvent]):
        self.config = config
        self.power_events = power_events
        self.time_points = np.arange(0, config.simulation_duration + 1, config.time_step)
        self.state_of_charge = np.zeros_like(self.time_points, dtype=float)
        self.battery_voltage = np.zeros_like(self.time_points, dtype=float)
        self.state_of_charge[0] = config.initial_state_of_charge
        self.battery_voltage[0] = config.min_voltage
        self.event_occurrences = {event.name: [] for event in power_events}
        
        # Validate solar panel data
        if config.battery_voltages is None or config.solar_panel_powers is None:
            raise ValueError("Battery voltage and solar panel power data must be provided")
        if len(config.battery_voltages) != len(config.solar_panel_powers):
            raise ValueError("Battery voltage and solar panel power arrays must have the same length")
    
    def get_solar_panel_power(self, battery_voltage: float) -> float:
        """Get solar panel power output for a given battery voltage."""
        # Interpolate power based on battery voltage
        return np.interp(battery_voltage, self.config.battery_voltages, self.config.solar_panel_powers)
    
    def voltage_from_soc(self, soc: float) -> float:
        """Convert state of charge to battery voltage."""
        # Simple linear model
        soc_fraction = soc / self.config.battery_capacity
        return self.config.min_voltage + (self.config.max_voltage - self.config.min_voltage) * soc_fraction
    
    def run_simulation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Run the battery simulation.
        
        Returns:
            Tuple containing time points, state of charge, battery voltage, and event occurrences
        """
        for i in range(1, len(self.time_points)):
            current_time = self.time_points[i]
            previous_charge = self.state_of_charge[i-1]
            previous_voltage = self.battery_voltage[i-1]
            
            # Calculate solar panel power based on current voltage
            solar_power = self.get_solar_panel_power(previous_voltage)
            
            # Calculate charging for this time step (convert power to energy)
            charging = solar_power * self.config.tumbling_efficiency_factor * (self.config.time_step / 3600.0)  # Convert seconds to hours
            new_charge = previous_charge + charging
            
            # Check for power events
            for event in self.power_events:
                if current_time % event.periodicity == 0:
                    # Calculate energy consumed by the event (convert power to energy)
                    event_energy = event.power_consumption * (event.duration / 3600.0)  # Convert seconds to hours
            
            # Ensure we don't exceed battery capacity
            new_charge = min(new_charge, self.config.battery_capacity)
            
            # Update state of charge and voltage
            self.state_of_charge[i] = new_charge
            self.battery_voltage[i] = self.voltage_from_soc(new_charge)
        
        return self.time_points, self.state_of_charge, self.battery_voltage, self.event_occurrences


def plot_results(time_points: np.ndarray, state_of_charge: np.ndarray, 
                 battery_voltage: np.ndarray, event_occurrences: dict, config: SimulationConfig,
                 power_events: List[PowerEvent]):
    """Plot the simulation results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot state of charge
    ax1.plot(time_points / 60, state_of_charge, 'b-', label='Battery State of Charge')
    ax1.axhline(y=config.battery_capacity, color='r', linestyle='--', 
                label='Battery Capacity')
    
    # Plot battery voltage
    ax2.plot(time_points / 60, battery_voltage, 'g-', label='Battery Voltage')
    ax2.axhline(y=config.max_voltage, color='r', linestyle='--', 
                label='Max Voltage')
    ax2.axhline(y=config.min_voltage, color='r', linestyle='--', 
                label='Min Voltage')
    
    # Mark event occurrences on both plots
    colors = ['m', 'c']
    for i, (event_name, times) in enumerate(event_occurrences.items()):
        if times:  # Only plot if there are occurrences
            color = colors[i % len(colors)]
            # Mark on state of charge plot
            ax1.scatter(np.array(times) / 60, 
                       np.zeros_like(times) + config.battery_capacity * 0.05, 
                       marker='v', color=color, label=f'{event_name} Events')
            # Mark on voltage plot
            ax2.scatter(np.array(times) / 60, 
                       np.zeros_like(times) + config.min_voltage + (config.max_voltage - config.min_voltage) * 0.05,
                       marker='v', color=color, label=f'{event_name} Events')
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('State of Charge (Watt-hours)')
    ax1.set_title('Spacecraft Battery Simulation')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Battery Voltage (V)')
    ax2.grid(True)
    ax2.legend()
    
    # Calculate energy consumption for each event using power_events values
    energy_text = f"Energy consumption per deployment:\n"
    for event in power_events:
        event_energy = event.power_consumption * (event.duration / 3600.0)  # Convert seconds to hours
        energy_text += f"{event.name}: {event_energy:.4f} Watt-hours ({event.power_consumption}W × {event.duration}s)\n"
    energy_text += f"Total battery capacity: {config.battery_capacity} Watt-hours"
    
    # Add explanatory text about energy consumption
    fig.text(0.02, 0.02, energy_text, fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig('battery_simulation.png')
    plt.show()


def analyze_results(state_of_charge: np.ndarray, event_occurrences: dict):
    """Analyze and print simulation results."""
    final_charge = state_of_charge[-1]
    charge_increase = state_of_charge[-1] - state_of_charge[0]
    
    print("\nSimulation Results:")
    print(f"Final battery state of charge: {final_charge:.2f} Watt-hours")
    print(f"Net charge increase: {charge_increase:.2f} Watt-hours")
    
    for event_name, times in event_occurrences.items():
        print(f"Number of {event_name} events: {len(times)}")


def main():
    """Main function to run the simulation."""
    # Default configuration
    config = SimulationConfig()
    
    # Define power events with default periodicities
    power_events = [
        PowerEvent(name="Antenna Deployment", power_consumption=2.5, duration=40.0, periodicity=30*60),  # duration and periodicity are in seconds
        PowerEvent(name="Solar Panel Deployment", power_consumption=3.6, duration=5.0, periodicity=60*60)  
    ]
    
    print(f"Running simulation with:")
    print(f"  Antenna deployment every {power_events[0].periodicity/60} minutes")
    print(f"  Solar panel deployment every {power_events[1].periodicity/60} minutes")
    print(f"  Duration: {config.simulation_duration/3600} hours")
    
    # Run simulation
    simulator = BatterySimulator(config, power_events)
    time_points, state_of_charge, battery_voltage, event_occurrences = simulator.run_simulation()
    
    # Analyze and visualize results
    analyze_results(state_of_charge, event_occurrences)
    plot_results(time_points, state_of_charge, battery_voltage, event_occurrences, config, power_events)


if __name__ == "__main__":
    main()
