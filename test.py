# Test script for functions in 263 Project 

from main import *
from concentration_functions import * 
from pressure_functions import * 
from qloss import * 

import numpy as np
import numpy.testing as nptest
from numpy.linalg import norm

testVal = 1.e-10

def test_dPdt():

	"""
    Test if the function dPdt is working properly by comparing it with a known result which includes edge case. 
    """

	# Test when all inputs are 0 
	test1a = dPdt(0.0, 0.0, [0,0,0,0],[0,0])
	assert((test1a - 0) < testVal)

    # Test when all inputs are 1
	test1b = dPdt(1,1,[1,1,1,1],[1,1])
	assert((test1b + 2) < testVal)

	# Test negative values
	test1c = dPdt(-1,-1,[-1,-1,-1,-1],[-1,-1])
	assert((test1c + 2) < testVal)

	# Test out string input 
	try:
		test1d = dPdt(10,'hi',[10,10,10,10],[10,10])
	except TypeError: 
		pass

	# Test out non-array input 
	try:
		test1e = dPdt(10, 10 , 5 ,[10,10])
	except TypeError: 
		pass

	# Test out incorrect variable type input 
	try:
		test1f = dPdt(1,1,[1,1,1,1],[1,1], 1.5)
	except TypeError: 
		pass

	# Test i value input
	test1g = dPdt(1,1,[1,1,1,1],[[1,2],[1,2]], 1)
	assert((test1g + 4) < testVal)

	# Test out of bounds i value 
	try:
		test1h = dPdt(1,1,[1,1,1,1],[[1,2],[1,2]], 2)
	except IndexError: 
		pass
	


def test_carbon_prime():

	"""
    Test if the function carbon_prime is working properly by comparing it with a known result which includes edge case. 
    """
	
	# Test for values of 0
	test2a = carbon_prime(0, 0, 0)
	assert((test2a - 0.03) < testVal)

	# Test when p = p0
	test2b = carbon_prime(1,1,1)
	assert((test2b - 0.03) < testVal)

	# Test when p > p0
	test2c = carbon_prime(1,2,0)		
	assert((test2c - 1) < testVal)

	# Test when p < p0
	test2d = carbon_prime(1,0,2)
	assert((test2d - 0.03) < testVal)

	# Test negative C input
	test2e = carbon_prime(-1,0,2)
	assert((test2e - 0.03) < testVal)

	# Test negative C input
	test2f = carbon_prime(-1,2,0)
	assert((test2f - 0.03) < testVal)

	# Test string input
	try:
		test2g = carbon_prime(1,'yes',2)
	except TypeError:
		pass

	# Test missing input
	try:
   		test2h = carbon_prime(1,2)
	except TypeError:
		pass



def test_dCdt():

	"""
    Test if the function dCdt is working properly by comparing it with a known result which includes edge case. 
    """

	# Test values of 0 (cannot divide by 0)
	try:
		test3a = dCdt(0.0,0.0,[0.0,0.0],[np.full(1,0),0.0,0.0,0.0,0.0,0.0,np.full(1,0)], 0)
	except ZeroDivisionError:
		pass

	# Test values of 1
	test3b = dCdt(1.0,1.0,[1.0,1.0],[np.full(1,1.0),1.0,1.0,1.0,1.0,1.0,np.full(1,1.0)], 0)
	assert((test3b - 0) < testVal)


	# Test negative values
	test3c = dCdt(-1.0,-1.0,[-1.0,-1.0],[np.full(1,-1),-1.0,-1.0,-1.0,-1.0,-1.0,np.full(1,-1)], 0)
	assert((test3c - 2.0) < testVal)

	# Test out of bounds index
	try:
		test3d = dCdt(1.0,1.0,[1.0,1.0],[np.full(1,1.0),1.0,1.0,1.0,1.0,1.0,np.full(1,1.0)], 10)
	except IndexError:
		pass

	# Test out string input 
	try:
		test3e = dCdt(1.0,'hi',[1.0,1.0],[np.full(1,1.0),1.0,1.0,1.0,1.0,1.0,np.full(1,1.0)], 10)
	except TypeError:
		pass

	# Test out non-array inputs for P and qco2
	try:
		test3f = dCdt(1.0,1.0,[1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0], 1)
	except TypeError:
		pass

	# Test out incorrect variable type input 
	try:
		test3g = dCdt(1.0,1.0,[1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0], 1.5)
	except TypeError:
		pass



def test_improved_euler_pressure():
	
	"""
    Test if the function improved_euler_pressure is working properly by comparing it with a known result which includes edge case. 
    """

	# Test one step 
	test4a= improved_euler_pressure(dPdt,0.0,1.0,1.0,0.0,[[0.0,0.0,0.0,0.0],[0.0,0.0]])
	assert((test4a[0] - 0) < testVal)
	assert((test4a[1] - 0) < testVal)

	# Test small steps 
	test4b = improved_euler_pressure(dPdt,0.0,1.0,0.1,0.0,[[0.0,0.0,0.0,0.0],[0.0,0.0]])
	sol4b = np.zeros(11)
	nptest.assert_almost_equal(test4b, sol4b)

	# Take no step 
	test4c= improved_euler_pressure(dPdt,0.0,0.0,1.0,0.0,[[0.0,0.0,0.0,0.0],[0.0,0.0]])
	assert((test4c - 0) < testVal)
	
	# Take one step with parameters of 1
	test4d= improved_euler_pressure(dPdt,0.0,1.0,1.0,0.0,[[1.0,1.0,1.0,1.0],[1.0,1.0]])
	assert((test4d[0] - 0) < testVal)
	assert((test4d[1] + 0.5) < testVal)

	# Take backwards step 
	try:
		test4e= improved_euler_pressure(dPdt,0.0,1.0,-1.0,1.0,[[1.0,1.0,1.0,1.0],[1.0,1.0]])
	except IndexError:
		pass
	
	# Input negative time values
	test4f= improved_euler_pressure(dPdt,0.0,-1.0,-4.0,0.0,[[1.0,1.0,1.0,1.0],[1.0,1.0]])
	assert((test4f[0] - 0) < testVal)
	assert((test4f[1] - 12) < testVal)

	# Test out string input 
	try:
		test4g= improved_euler_pressure(dPdt,'hi',1.0,1.0,0.0,[[1.0,1.0,1.0,1.0],[1.0,1.0]])
	except TypeError:
		pass

	# Test out non-array inputs 
	try:
		test4h= improved_euler_pressure(dPdt,0.0,1.0,1.0,0.0,[1.0],[1.0,1.0])
	except TypeError:
		pass

	# Test out incorrect variable type input 
	try:
		test4i= improved_euler_pressure(dPdt,0.0,[1.0,2.0],1.0,0.0,[[1.0,1.0,1.0,1.0],[1.0,1.0]])
	except TypeError:
		pass

	# Test out step variable larger than range
	try:
		test4j= improved_euler_pressure(dPdt,0.0,1.0,2.0,5.0,[[1.0,1.0,1.0,1.0],[1.0,1.0]])
	except TypeError:
		pass



def test_improved_euler_concentration():
	
	"""
    Test if the function improved_euler_concentration is working properly by comparing it with a known result which includes edge case. 
    """

	# Test one step with zero values
	try:
		test5a= improved_euler_concentration(dCdt,0.0,1.0,1.0,0.0,[[0.0,0.0],[np.full(1,0),0.0,0.0,0.0,0.0,0.0,np.full(1,0)]])
	except ZeroDivisionError:
		pass
	
	# Test one euler step
	test5b= improved_euler_concentration(dCdt,0.0,1.0,1.0,0.0,[[1.0,1.0],[np.full(1,1.0),1.0,1.0,1.0,1.0,1.0,np.full(1,1.0)]])
	assert((test5b[0] - 0) < testVal)
	assert((test5b[1] - 0) < testVal)

	# Test one euler step with larger parameters
	test5c= improved_euler_concentration(dCdt,0.0,1.0,1.0,0.0,[[10.0,10.0],[np.full(1,10.0),10.0,10.0,10.0,10.0,10.0,np.full(1,10.0)]])
	assert((test5c[0] - 0) < testVal)
	assert((test5c[1] + 454.5) < testVal)


	# Take no step 
	try:
		test5d= improved_euler_concentration(dCdt,0.0,1.0,0.0,0.0,[[10.0,10.0],[np.full(1,10.0),10.0,10.0,10.0,10.0,10.0,np.full(1,10.0)]])
	except ZeroDivisionError:
		pass
	
	# Take backwards step 
	try:
		test5e= improved_euler_concentration(dCdt,0.0,1.0,-1.0,0.0,[[10.0,10.0],[np.full(1,10.0),10.0,10.0,10.0,10.0,10.0,np.full(1,10.0)]])
	except IndexError:
		pass

	# Test string input
	try:
		test5f= improved_euler_concentration(dCdt,0.0,'hi',-1.0,0.0,[[10.0,10.0],[np.full(1,10.0),10.0,10.0,10.0,10.0,10.0,np.full(1,10.0)]])
	except TypeError:
		pass

	# Test non-array input
	try:
		test5g= improved_euler_concentration(dCdt,0.0,1.0,-1.0,0.0,[[10.0,10.0],[1,10.0,10.0,10.0,10.0,10.0,1.0]])
	except IndexError:
		pass

	# Test incorrect variable input
	try:
		test5h= improved_euler_concentration(dCdt,0.0,1.0,[1.0,2.0],0.0,[[10.0,10.0],[1,10.0,10.0,10.0,10.0,10.0,1.0]])
	except TypeError:
		pass

	# Test out step variable larger than range
	try:
		test5i= improved_euler_concentration(dCdt,0.0,1.0,5.0,0.0,[[10.0,10.0],[1,10.0,10.0,10.0,10.0,10.0,1.0]])
	except TypeError:
		pass



def test_dq_lossdt():
	
	"""
    Test if the function dq_lossdt is working properly by comparing it with a known result which includes edge case. 
    """
	
	# Test when all inputs are 0 
	try:
		test6a= dq_lossdt(0.0,0.0,[0.0,np.full(10,0.0),np.full(10,0.0),0.0,0.0], i=0)
	except RuntimeWarning:
		pass
	
	# Test when all inputs are 1 
	test6b= dq_lossdt(1.0,1.0,[1.0,np.full(10,1.0),np.full(10,1.0),1.0,1.0], i=0)
	assert((test6b - 0) < testVal)

	# Test with negative inputs
	try:
		test6c= dq_lossdt(1.0,1.0,[-1.0,np.full(10,-1.0),np.full(10,-1.0),-1.0,-1.0], i=0)
	except RuntimeWarning:
		pass
	
	# Test string input
	try:
		test6d= dq_lossdt(0.0,'hi',[0.0,np.full(10,0.0),np.full(10,0.0),0.0,0.0], i=0)
	except TypeError:
		pass

	# Test non-array input
	try:
		test6e= dq_lossdt(0.0,0.0,[5.0,5.0,0.0,0.0], i=0)
	except ValueError:
		pass

	# Test out of bounds i index
	try:
		test6f= dq_lossdt(0.0,0.0,[0.0,np.full(10,0.0),np.full(10,0.0),0.0,0.0], i=12)
	except IndexError:
		pass



def test_carbon_prime_q():
	
	"""
    Test if the function carbon_prime_q is working properly by comparing it with a known result which includes edge case. 
    """

	# Test for values of 0
	test7a = carbon_prime_q(0, 0, 0)
	assert((test7a - 0.03) < testVal)

	# Test when p = p0
	test7b = carbon_prime_q(1,1,1)
	assert((test7b - 0.03) < testVal)

	# Test when p > p0
	test7c = carbon_prime_q(1,2,0)		
	assert((test7c - 1) < testVal)

	# Test when p < p0
	test7d = carbon_prime_q(1,0,2)
	assert((test7d - 0.03) < testVal)

	# Test negative C input
	test7e = carbon_prime_q(-1,0,2)
	assert((test7e - 0.03) < testVal)

	# Test negative C input
	test7f = carbon_prime_q(-1,2,0)
	assert((test7f - 0.03) < testVal)

	# Test string input
	try:
		test7g = carbon_prime_q(1,'yes',2)
	except TypeError:
		pass

	# Test missing input
	try:
   		test7h = carbon_prime_q(1,2)
	except TypeError:
		pass



def test_improved_euler_q_loss():
	
	"""
    Test if the function improved_euler_q_loss is working properly by comparing it with a known result which includes edge case. 
    """
	
	# Test one step with zero values
	try:
		test8a= improved_euler_q_loss(dq_lossdt,0.0,1.0,1.0,0.0,[0.0,np.full(10,0.0),np.full(10,0.0),0.0,0.0])
	except ZeroDivisionError:
		pass
	
	# Test one euler step
	test8b= improved_euler_q_loss(dq_lossdt,0.0,1.0,1.0,0.0,[5.0,np.full(10,5.0),np.full(10,5.0),5.0,5.0])
	assert((test8b[0] - 0) < testVal)
	assert((test8b[1] - 0) < testVal)

	# Test one euler step with varied parameters
	test8c= improved_euler_q_loss(dq_lossdt,0.0,1.0,0.5,0.0,[1.0,np.full(10,2.0),np.full(10,3.0),4.0,5.0])
	assert((test8b[0] - 0) < testVal)
	assert((test8c[1] - 1.875) < testVal)
	assert((test8c[2] - 3.75) < testVal)
	

	# Take no step 
	try:
		test8d= improved_euler_q_loss(dq_lossdt,0.0,1.0,0.0,0.0,[1.0,np.full(10,2.0),np.full(10,3.0),4.0,5.0])
	except ZeroDivisionError:
		pass
	
	# Take backwards step 
	try:
		test8e= improved_euler_q_loss(dq_lossdt,0.0,1.0,-1.0,0.0,[1.0,np.full(10,2.0),np.full(10,3.0),4.0,5.0])
	except IndexError:
		pass

	# Test string input
	try:
		test8f= improved_euler_q_loss(dq_lossdt,0.0,'hi',0.0,0.0,[1.0,np.full(10,2.0),np.full(10,3.0),4.0,5.0])
	except TypeError:
		pass


	# Test incorrect variable input
	try:
		test8g= improved_euler_q_loss(dq_lossdt,0.0,1.0,1.0,1.0,[1.0,np.full(10,2.0),np.full(10,3.0),4.0,5.0])
	except TypeError:
		pass

	# Test out step variable larger than range
	try:
		test8h= improved_euler_q_loss(dq_lossdt,0.0,1.0,5.0,0.0,[1.0,np.full(10,2.0),np.full(10,3.0),4.0,5.0])
	except TypeError:
		pass




def main():
	test_dPdt()
	test_carbon_prime()
	test_dCdt()
	test_improved_euler_pressure()
	test_improved_euler_concentration()
	test_dq_lossdt()
	test_carbon_prime_q()
	test_improved_euler_q_loss()


if __name__=="__main__":
	main()