"""
Airline domain tools wrapped for Strands.
Creates closure-based tool functions that maintain database state.
"""

import json
from typing import List, Dict, Any, Optional
from copy import deepcopy

from strands import tool

# We'll need to import Tau-2 data models
# Adjust import path based on your tau2 installation
try:
    from tau2.domains.airline.data_model import (
        FlightDB, Flight, User, Reservation, 
        CabinClass, FlightType, Insurance,
        Passenger, Payment, FlightInfo
    )
    from tau2.domains.airline.tools import AirlineTools
    TAU2_AVAILABLE = True
except ImportError:
    TAU2_AVAILABLE = False
    print("Warning: tau2 not available, airline tools will not work")


class AirlineToolsWrapper:
    """
    Wrapper that creates Strands-compatible tools from AirlineTools.
    Maintains database state across tool calls.
    """
    
    def __init__(self, db_dict: Optional[Dict] = None, db_path: Optional[str] = None):
        """
        Initialize with either a database dict or path to db.json.
        """
        if not TAU2_AVAILABLE:
            raise ImportError("tau2 package not available")
        
        if db_dict:
            self.db = FlightDB.model_validate(db_dict)
        elif db_path:
            self.db = FlightDB.load(db_path)
        else:
            # Load default db
            from tau2.domains.airline.utils import AIRLINE_DB_PATH
            self.db = FlightDB.load(AIRLINE_DB_PATH)
        
        self.toolkit = AirlineTools(self.db)
        self._initial_db_state = deepcopy(db_dict) if db_dict else None
    
    def reset(self, db_dict: Optional[Dict] = None):
        """Reset database to initial state."""
        if db_dict:
            self.db = FlightDB.model_validate(db_dict)
        elif self._initial_db_state:
            self.db = FlightDB.model_validate(deepcopy(self._initial_db_state))
        else:
            from tau2.domains.airline.utils import AIRLINE_DB_PATH
            self.db = FlightDB.load(AIRLINE_DB_PATH)
        
        self.toolkit = AirlineTools(self.db)
    
    def get_tools(self) -> List:
        """Get list of Strands-compatible tool functions."""
        
        toolkit = self.toolkit
        
        @tool
        def get_user_details(user_id: str) -> str:
            """
            Get the details of a user, including their reservations.
            
            Args:
                user_id: The user ID, such as 'sara_doe_496'.
            
            Returns:
                JSON string with user details including name, address, email, 
                payment methods, saved passengers, membership level, and reservations.
            """
            try:
                result = toolkit.get_user_details(user_id)
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def get_reservation_details(reservation_id: str) -> str:
            """
            Get the details of a reservation.
            
            Args:
                reservation_id: The reservation ID, such as '8JX2WO'.
            
            Returns:
                JSON string with reservation details including flights, passengers,
                cabin class, baggage, insurance, and payment history.
            """
            try:
                result = toolkit.get_reservation_details(reservation_id)
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def search_direct_flight(origin: str, destination: str, date: str) -> str:
            """
            Search for direct flights between two cities on a specific date.
            
            Args:
                origin: The origin city airport in three letters, such as 'JFK'.
                destination: The destination city airport in three letters, such as 'LAX'.
                date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.
            
            Returns:
                JSON string with list of available direct flights including 
                flight numbers, times, prices, and available seats.
            """
            try:
                result = toolkit.search_direct_flight(origin, destination, date)
                return json.dumps([r.model_dump() for r in result], indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def search_onestop_flight(origin: str, destination: str, date: str) -> str:
            """
            Search for one-stop flights between two cities on a specific date.
            
            Args:
                origin: The origin city airport in three letters, such as 'JFK'.
                destination: The destination city airport in three letters, such as 'LAX'.
                date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.
            
            Returns:
                JSON string with list of one-stop flight pairs.
            """
            try:
                result = toolkit.search_onestop_flight(origin, destination, date)
                return json.dumps([[r[0].model_dump(), r[1].model_dump()] for r in result], indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def list_all_airports() -> str:
            """
            Returns a list of all available airports.
            
            Returns:
                JSON string with list of airport codes and city names.
            """
            try:
                result = toolkit.list_all_airports()
                return json.dumps([r.model_dump() for r in result], indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def get_flight_status(flight_number: str, date: str) -> str:
            """
            Get the status of a flight.
            
            Args:
                flight_number: The flight number, such as 'HAT001'.
                date: The date of the flight in the format 'YYYY-MM-DD'.
            
            Returns:
                Status string: 'available', 'on time', 'delayed', 'cancelled', 'landed', or 'flying'.
            """
            try:
                result = toolkit.get_flight_status(flight_number, date)
                return result
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def book_reservation(
            user_id: str,
            origin: str,
            destination: str,
            flight_type: str,
            cabin: str,
            flights: str,
            passengers: str,
            payment_methods: str,
            total_baggages: int,
            nonfree_baggages: int,
            insurance: str
        ) -> str:
            """
            Book a reservation.
            
            Args:
                user_id: The ID of the user, such as 'sara_doe_496'.
                origin: The IATA code for the origin city, such as 'SFO'.
                destination: The IATA code for the destination city, such as 'JFK'.
                flight_type: Type of flight - 'one_way' or 'round_trip'.
                cabin: Cabin class - 'basic_economy', 'economy', or 'business'.
                flights: JSON string array of flight objects with 'flight_number' and 'date'.
                passengers: JSON string array of passenger objects with 'first_name', 'last_name', 'dob'.
                payment_methods: JSON string array of payment objects with 'payment_id' and 'amount'.
                total_baggages: Total number of baggage items.
                nonfree_baggages: Number of paid baggage items.
                insurance: Whether to include insurance - 'yes' or 'no'.
            
            Returns:
                JSON string with the created reservation details.
            """
            try:
                flights_list = json.loads(flights) if isinstance(flights, str) else flights
                passengers_list = json.loads(passengers) if isinstance(passengers, str) else passengers
                payments_list = json.loads(payment_methods) if isinstance(payment_methods, str) else payment_methods
                
                result = toolkit.book_reservation(
                    user_id=user_id,
                    origin=origin,
                    destination=destination,
                    flight_type=flight_type,
                    cabin=cabin,
                    flights=flights_list,
                    passengers=passengers_list,
                    payment_methods=payments_list,
                    total_baggages=total_baggages,
                    nonfree_baggages=nonfree_baggages,
                    insurance=insurance
                )
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def cancel_reservation(reservation_id: str) -> str:
            """
            Cancel the whole reservation. Only allowed for:
            - Reservations within 24 hours of booking
            - Business class reservations
            - Reservations with travel insurance
            
            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.
            
            Returns:
                JSON string with the updated reservation showing cancelled status.
            """
            try:
                result = toolkit.cancel_reservation(reservation_id)
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def update_reservation_flights(
            reservation_id: str,
            cabin: str,
            flights: str,
            payment_id: str
        ) -> str:
            """
            Update the flight information of a reservation.
            Cannot change basic economy reservations. Cannot change origin/destination.
            
            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.
                cabin: The cabin class - 'basic_economy', 'economy', or 'business'.
                flights: JSON string array of ALL flight objects in the new reservation.
                payment_id: Payment method ID for any price difference, such as 'credit_card_7815826'.
            
            Returns:
                JSON string with the updated reservation details.
            """
            try:
                flights_list = json.loads(flights) if isinstance(flights, str) else flights
                
                result = toolkit.update_reservation_flights(
                    reservation_id=reservation_id,
                    cabin=cabin,
                    flights=flights_list,
                    payment_id=payment_id
                )
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def update_reservation_passengers(reservation_id: str, passengers: str) -> str:
            """
            Update the passenger information of a reservation.
            Number of passengers must remain the same.
            
            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.
                passengers: JSON string array of passenger objects with 'first_name', 'last_name', 'dob'.
            
            Returns:
                JSON string with the updated reservation details.
            """
            try:
                passengers_list = json.loads(passengers) if isinstance(passengers, str) else passengers
                
                result = toolkit.update_reservation_passengers(
                    reservation_id=reservation_id,
                    passengers=passengers_list
                )
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def update_reservation_baggages(
            reservation_id: str,
            total_baggages: int,
            nonfree_baggages: int,
            payment_id: str
        ) -> str:
            """
            Update the baggage information of a reservation.
            Can add bags but cannot remove bags.
            
            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.
                total_baggages: The updated total number of baggage items.
                nonfree_baggages: The updated number of paid baggage items.
                payment_id: Payment method ID for baggage fees, such as 'credit_card_7815826'.
            
            Returns:
                JSON string with the updated reservation details.
            """
            try:
                result = toolkit.update_reservation_baggages(
                    reservation_id=reservation_id,
                    total_baggages=total_baggages,
                    nonfree_baggages=nonfree_baggages,
                    payment_id=payment_id
                )
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def send_certificate(user_id: str, amount: int) -> str:
            """
            Send a certificate/voucher to a user as compensation.
            Use for flight delays, cancellations, or service issues.
            
            Args:
                user_id: The ID of the user, such as 'sara_doe_496'.
                amount: The amount of the certificate in dollars.
            
            Returns:
                Confirmation message with certificate details.
            """
            try:
                result = toolkit.send_certificate(user_id, amount)
                return result
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def calculate(expression: str) -> str:
            """
            Calculate the result of a mathematical expression.
            
            Args:
                expression: The mathematical expression, such as '2 + 2' or '(100 - 50) * 2'.
            
            Returns:
                The calculated result as a string.
            """
            try:
                result = toolkit.calculate(expression)
                return result
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def transfer_to_human_agents(summary: str) -> str:
            """
            Transfer the user to a human agent. Only use when:
            - The user explicitly asks for a human agent
            - The issue cannot be resolved with available tools
            
            Args:
                summary: A summary of the user's issue for the human agent.
            
            Returns:
                Confirmation of transfer.
            """
            try:
                result = toolkit.transfer_to_human_agents(summary)
                return result
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def done(response: str) -> str:
            """
            Call this when you have completed all necessary actions for the customer's issue.
            
            Args:
                response: A summary of what actions were taken to resolve the issue.
            
            Returns:
                Confirmation that the task is complete.
            """
            return f"Task completed: {response}"
        
        return [
            get_user_details,
            get_reservation_details,
            search_direct_flight,
            search_onestop_flight,
            list_all_airports,
            get_flight_status,
            book_reservation,
            cancel_reservation,
            update_reservation_flights,
            update_reservation_passengers,
            update_reservation_baggages,
            send_certificate,
            calculate,
            transfer_to_human_agents,
            done,
        ]


def create_airline_tools(db_dict: Optional[Dict] = None) -> tuple:
    """
    Factory function to create airline tools.
    Returns (tools_list, wrapper_instance) so the wrapper can be reset between episodes.
    """
    wrapper = AirlineToolsWrapper(db_dict=db_dict)
    return wrapper.get_tools(), wrapper