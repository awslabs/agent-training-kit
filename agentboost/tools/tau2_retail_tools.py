"""
Retail domain tools wrapped for Strands.
Creates closure-based tool functions that maintain database state.
"""

import json
from typing import List, Dict, Any, Optional
from copy import deepcopy

from strands import tool

try:
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.tools import RetailTools
    TAU2_AVAILABLE = True
except ImportError:
    TAU2_AVAILABLE = False
    print("Warning: tau2 not available, retail tools will not work")


class RetailToolsWrapper:
    """
    Wrapper that creates Strands-compatible tools from RetailTools.
    Maintains database state across tool calls.
    """
    
    def __init__(self, db_dict: Optional[Dict] = None, db_path: Optional[str] = None):
        if not TAU2_AVAILABLE:
            raise ImportError("tau2 package not available")
        
        if db_dict:
            self.db = RetailDB.model_validate(db_dict)
        elif db_path:
            self.db = RetailDB.load(db_path)
        else:
            from tau2.domains.retail.utils import RETAIL_DB_PATH
            self.db = RetailDB.load(RETAIL_DB_PATH)
        
        self.toolkit = RetailTools(self.db)
        self._initial_db_state = deepcopy(db_dict) if db_dict else None
    
    def reset(self, db_dict: Optional[Dict] = None):
        """Reset database to initial state."""
        if db_dict:
            self.db = RetailDB.model_validate(db_dict)
        elif self._initial_db_state:
            self.db = RetailDB.model_validate(deepcopy(self._initial_db_state))
        else:
            from tau2.domains.retail.utils import RETAIL_DB_PATH
            self.db = RetailDB.load(RETAIL_DB_PATH)
        
        self.toolkit = RetailTools(self.db)
    
    def get_tools(self) -> List:
        """Get list of Strands-compatible tool functions."""
        
        toolkit = self.toolkit
        
        @tool
        def find_user_id_by_email(email: str) -> str:
            """
            Find user id by email. Preferred method for user lookup.
            
            Args:
                email: The email of the user, such as 'john@example.com'.
            
            Returns:
                The user ID if found, or error message.
            """
            try:
                result = toolkit.find_user_id_by_email(email)
                return result
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def find_user_id_by_name_zip(first_name: str, last_name: str, zip_code: str) -> str:
            """
            Find user id by first name, last name, and zip code.
            Use only if email lookup fails.
            
            Args:
                first_name: The first name, such as 'John'.
                last_name: The last name, such as 'Doe'.
                zip_code: The zip code, such as '12345'.
            
            Returns:
                The user ID if found, or error message.
            """
            try:
                result = toolkit.find_user_id_by_name_zip(first_name, last_name, zip_code)
                return result
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def get_user_details(user_id: str) -> str:
            """
            Get the details of a user, including their orders.
            
            Args:
                user_id: The user ID, such as 'sara_doe_496'.
            
            Returns:
                JSON string with user details including name, address, email,
                payment methods, and order IDs.
            """
            try:
                result = toolkit.get_user_details(user_id)
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def get_order_details(order_id: str) -> str:
            """
            Get the status and details of an order.
            
            Args:
                order_id: The order ID with '#' prefix, such as '#W0000000'.
            
            Returns:
                JSON string with order details including items, status,
                fulfillment info, and payment history.
            """
            try:
                result = toolkit.get_order_details(order_id)
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def get_product_details(product_id: str) -> str:
            """
            Get the inventory details of a product.
            
            Args:
                product_id: The product ID, such as '6086499569'.
                           Note: product_id is different from item_id.
            
            Returns:
                JSON string with product name and all variants with 
                their options, availability, and prices.
            """
            try:
                result = toolkit.get_product_details(product_id)
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def list_all_product_types() -> str:
            """
            List the name and product id of all product types.
            There are 50 product types in the store.
            
            Returns:
                JSON string mapping product names to product IDs.
            """
            try:
                result = toolkit.list_all_product_types()
                return result
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def cancel_pending_order(order_id: str, reason: str) -> str:
            """
            Cancel a pending order. Only pending orders can be cancelled.
            Must get explicit user confirmation before calling.
            
            Args:
                order_id: The order ID, such as '#W0000000'.
                reason: Reason for cancellation - 'no longer needed' or 'ordered by mistake'.
            
            Returns:
                JSON string with updated order details showing cancelled status.
            """
            try:
                result = toolkit.cancel_pending_order(order_id, reason)
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def modify_pending_order_items(
            order_id: str,
            item_ids: str,
            new_item_ids: str,
            payment_method_id: str
        ) -> str:
            """
            Modify items in a pending order to different variants of the SAME product.
            Can only be called once per order. Must get user confirmation first.
            
            Args:
                order_id: The order ID, such as '#W0000000'.
                item_ids: JSON array of item IDs to modify, such as '["1008292230"]'.
                new_item_ids: JSON array of new item IDs (same order as item_ids).
                payment_method_id: Payment method for price difference.
            
            Returns:
                JSON string with updated order details.
            """
            try:
                item_ids_list = json.loads(item_ids) if isinstance(item_ids, str) else item_ids
                new_item_ids_list = json.loads(new_item_ids) if isinstance(new_item_ids, str) else new_item_ids
                
                result = toolkit.modify_pending_order_items(
                    order_id=order_id,
                    item_ids=item_ids_list,
                    new_item_ids=new_item_ids_list,
                    payment_method_id=payment_method_id
                )
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def modify_pending_order_address(
            order_id: str,
            address1: str,
            address2: str,
            city: str,
            state: str,
            country: str,
            zip_code: str
        ) -> str:
            """
            Modify the shipping address of a pending order.
            Must get user confirmation first.
            
            Args:
                order_id: The order ID, such as '#W0000000'.
                address1: Primary address line, such as '123 Main St'.
                address2: Secondary address line, such as 'Apt 1' or empty string.
                city: City name, such as 'San Francisco'.
                state: State code, such as 'CA'.
                country: Country name, such as 'USA'.
                zip_code: Postal code, such as '12345'.
            
            Returns:
                JSON string with updated order details.
            """
            try:
                result = toolkit.modify_pending_order_address(
                    order_id=order_id,
                    address1=address1,
                    address2=address2,
                    city=city,
                    state=state,
                    country=country,
                    zip=zip_code
                )
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def modify_pending_order_payment(order_id: str, payment_method_id: str) -> str:
            """
            Modify the payment method of a pending order.
            Must get user confirmation first.
            
            Args:
                order_id: The order ID, such as '#W0000000'.
                payment_method_id: New payment method ID.
            
            Returns:
                JSON string with updated order details.
            """
            try:
                result = toolkit.modify_pending_order_payment(
                    order_id=order_id,
                    payment_method_id=payment_method_id
                )
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def modify_user_address(
            user_id: str,
            address1: str,
            address2: str,
            city: str,
            state: str,
            country: str,
            zip_code: str
        ) -> str:
            """
            Modify the default address of a user.
            Must get user confirmation first.
            
            Args:
                user_id: The user ID, such as 'sara_doe_496'.
                address1: Primary address line.
                address2: Secondary address line or empty string.
                city: City name.
                state: State code.
                country: Country name.
                zip_code: Postal code.
            
            Returns:
                JSON string with updated user details.
            """
            try:
                result = toolkit.modify_user_address(
                    user_id=user_id,
                    address1=address1,
                    address2=address2,
                    city=city,
                    state=state,
                    country=country,
                    zip=zip_code
                )
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def return_delivered_order_items(
            order_id: str,
            item_ids: str,
            payment_method_id: str
        ) -> str:
            """
            Return items from a delivered order. 
            For delivered orders, return or exchange can only be done once.
            Must get user confirmation first.
            
            Args:
                order_id: The order ID, such as '#W0000000'.
                item_ids: JSON array of item IDs to return.
                payment_method_id: Payment method for refund.
            
            Returns:
                JSON string with updated order showing 'return requested' status.
            """
            try:
                item_ids_list = json.loads(item_ids) if isinstance(item_ids, str) else item_ids
                
                result = toolkit.return_delivered_order_items(
                    order_id=order_id,
                    item_ids=item_ids_list,
                    payment_method_id=payment_method_id
                )
                return json.dumps(result.model_dump(), indent=2, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def exchange_delivered_order_items(
            order_id: str,
            item_ids: str,
            new_item_ids: str,
            payment_method_id: str
        ) -> str:
            """
            Exchange items in a delivered order for different variants of the SAME product.
            For delivered orders, return or exchange can only be done once.
            Must get user confirmation first.
            
            Args:
                order_id: The order ID, such as '#W0000000'.
                item_ids: JSON array of item IDs to exchange.
                new_item_ids: JSON array of new item IDs (same order, same products).
                payment_method_id: Payment method for price difference.
            
            Returns:
                JSON string with updated order showing 'exchange requested' status.
            """
            try:
                item_ids_list = json.loads(item_ids) if isinstance(item_ids, str) else item_ids
                new_item_ids_list = json.loads(new_item_ids) if isinstance(new_item_ids, str) else new_item_ids
                
                result = toolkit.exchange_delivered_order_items(
                    order_id=order_id,
                    item_ids=item_ids_list,
                    new_item_ids=new_item_ids_list,
                    payment_method_id=payment_method_id
                )
                return json.dumps(result.model_dump(), indent=2, default=str)
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
            find_user_id_by_email,
            find_user_id_by_name_zip,
            get_user_details,
            get_order_details,
            get_product_details,
            list_all_product_types,
            cancel_pending_order,
            modify_pending_order_items,
            modify_pending_order_address,
            modify_pending_order_payment,
            modify_user_address,
            return_delivered_order_items,
            exchange_delivered_order_items,
            calculate,
            transfer_to_human_agents,
            done,
        ]


def create_retail_tools(db_dict: Optional[Dict] = None) -> tuple:
    """
    Factory function to create retail tools.
    Returns (tools_list, wrapper_instance) so the wrapper can be reset between episodes.
    """
    wrapper = RetailToolsWrapper(db_dict=db_dict)
    return wrapper.get_tools(), wrapper