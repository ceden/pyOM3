
"""
 Setup of the model and base class.

 Contains only a dummy function loading all parameters
 given on initialisation.

"""

class setup():

      def __init__(self,parameter = {}):
         """
         Add parameter to instance, dictionary parameter contains
         various model parameters. No checks.
         """
         for key in parameter: setattr(self,key,parameter[key])
         return
