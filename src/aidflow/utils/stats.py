"""
Combinatorial Optimization and Statistical Utility Layer.

This module provides high-performance recursive implementations of fundamental 
combinatorial algorithms, specifically designed to support the exhaustive search 
space required for ML feature selection and hyperparameter optimization. 
"""

class Stats:
    """
    Utility class providing recursive implementations for common combinatorial operations.
    
    This class offers native Python alternatives to `itertools` for scenarios requiring
    custom recursion logic, debugging, or educational demonstrations of combinatorial 
    algorithms (Combinations, Permutations, Cartesian Products).
    
    Methods:
        combination: Generate combinations of a specific length from a list.
        permutation: Generate all permutations of a list.
        product: Compute the Cartesian product of multiple lists.
    """

    @classmethod
    def combination(cls, X:list[str|int|float|bool], length:int) -> list:
        """
        Recursively generates all combinations of a specific length from a list.
        
        Unlike permutations, order does not matter, and elements are not repeated
        within a single combination. The function preserves the original order of
        elements as they appear in the input list.
        
        Parameters
        ----------
        X : list
            The source list of elements to combine.
        length : int
            The desired length of each combination.
            
        Returns
        -------
        list of list
            A list containing all possible combinations as sub-lists.
            Returns `[[]]` if length is 0, or `[]` if X is empty.
            
        Example
        -------
        >>> Stats.combination(['A', 'B', 'C', 'D'], 2)
        [['A', 'B'], 
         ['A', 'C'], 
         ['A', 'D'], 
         ['B', 'C'], 
         ['B', 'D'], 
         ['C', 'D']]
        """        
        if length == 0:
            return [[]]
        if not X:
            return []
        C = []
        for i in range(len(X)):
            e0 = [X[i]]
            er = X[i+1:]
            for sub in cls.combination(er, length - 1):
                C.append(e0 + sub)
        return C

    @classmethod
    def permutation(cls, X:list[str|int|float|bool]) -> list:
        """
        Recursively generates all permutations of a list.
        
        Order matters in permutations, and every element is used exactly once 
        in each permutation. The total number of results is N! (N factorial).
        
        Parameters
        ----------
        X : list
            The list of elements to permute.
            
        Returns
        -------
        list of list
            A list containing all possible permutations as sub-lists.
            
        Example
        -------
        >>> Stats.permutation(['A', 'B', 'C'])
        [['A', 'B', 'C'],
         ['A', 'C', 'B'],
         ['B', 'A', 'C'],
         ['B', 'C', 'A'],
         ['C', 'A', 'B'],
         ['C', 'B', 'A']]
        """        
        if len(X) == 1:
            return [X]    
        P = []
        for i in range(len(X)):
            e0 = [X[i]]
            er = X[:i] + X[i+1:]
            for sub in cls.permutation(er):
                P.append(e0 + sub)
        return P

    @classmethod
    def product(cls, lists:list[list]) -> list:
        """
        Compute the Cartesian product of multiple lists recursively.
        
        This is a custom implementation equivalent to `itertools.product`, 
        returning the result as a list of lists. It iterates through every 
        element of the first list and combines it with every result from the 
        product of the remaining lists.
        
        Parameters
        ----------
        lists : list of list
            A list where each element is a list of items to combine.
            
        Returns
        -------
        list of list
            A list containing all possible combinations (Cartesian product) 
            as sub-lists. Returns an empty list if `lists` is empty.
            
        Example
        -------
        >>> Stats.product([['A', 'B', 'C'], [1, 2]])
        [['A', 1], ['A', 2], ['B', 1], ['B', 2], ['C', 1], ['C', 2]]
        """
        if not lists:
            return []
        
        if len(lists) == 1:
            return [[item] for item in lists[0]]
            
        first = lists[0]
        rest  = lists[1:]
        rest_product = cls.product(rest)
        
        result = []
        for item in first:
            for combo in rest_product:
                result.append([item] + combo)
        return result