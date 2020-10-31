import sys
import os
sys.path.append('../')
os.getcwd()

import unitest




from src.GridWord import *

if __name__ == '__main__':
    
    unitest.main()
    
    
class test_GridWord(unitest.TestCase):
    
    def setUp(self):
        
        return None
    
    
    def test_get_trace(self):
        
        self.
