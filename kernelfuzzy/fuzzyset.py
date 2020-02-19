"""

    Class FuzzySet

"""

class FuzzySet:

    _elements      = None  # Elements of the fuzzy set
    _elements_type = None  # Type of elements
    _membership_function            = None  # Membership function
    _membership_degrees            = None  # Membership degrees
    _membership_function_params        = None  # Membership function parameters

    def __init__(self, elements=None, membership_degrees=None, membership_function=None, membership_function_params=None):

        #first type: empty fuzzy set
        if elements is None and  membership_degrees is None and membership_function is None and membership_function_params is None:
            self._elements = None
            self._elements_type = None
            self._membership_degrees = membership_degrees = None
            self._membership_function_params = membership_function_params = None
            self._membership_function = None

        #second type: elements and membership degrees  are given but not the membership function
        if elements is not None:
            self._elements = elements
            if isinstance(self._elements, (float, int)):
                self._elements_type = type(elements)
            else:
                self._elements_type = type(elements[0])

        if elements is not None and membership_function is None:
            self._membership_degrees=membership_degrees

        # third type: elements and membership function  are given, then the membership degrees are estimated
        if elements is not None and membership_degrees is None:
            self._membership_function_params = membership_function_params
            self._membership_function = membership_function
            self._membership_degrees = self._membership_function(self._elements, *self._membership_function_params)

        # Fourth type: only # Membership function parameters

        if elements is None and membership_degrees is None and membership_function is None and membership_function_params is not None:
            self._membership_function_params = membership_function_params

    def set_membership_degree(self, membership_degrees):

        """

        Set the membership degrees

        Input:
            md: (Type: float)

        """

        self._membership_degrees = membership_degrees

    def get_elements(self):

        """

        Returns the set

        """

        return self._elements

    def get_membership_function_params(self):
        return self._membership_function_params

    def get_membership_function(self):

        """

        Returns the membership function

        """
        
        return self._membership_function

    def get_pair(self):
        
        """

        Returns the pair (_elements, _md) elements and membership degree

        """
        
        if  isinstance(self._elements,  (float, int)) and isinstance(self._membership_degrees, (float, int)):
            return list(zip(list([self._elements]), list([self._membership_degrees])))
        else:
            return list(zip(self._elements, self._membership_degrees))

    def get_membership_degrees(self):
        
        """

        Returns the membership degrees

        """
        
        return self._membership_degrees

    def show(self):

        """

        Print in the stdout the all the contents of the class, for debugging

        """
        print('\n')
        print("Fuzzy set: {}".format(self.get_pair()))
        print('\n')
        print('Properties:')

        print("(_elements)      \n", self._elements, "\n")   
        print("(_elements_type) \n", self._elements_type, "\n")   
        print("(_membership_function)            \n", self._membership_function, "\n")
        print("(_membership_degrees)            \n", self._membership_degrees, "\n")
        print("(_membership_function_params)        \n", self._membership_function_params, "\n")
