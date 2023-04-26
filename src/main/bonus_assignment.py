#Bonus Assignment
#Vallesia Pierre Louis


import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)
#1. The number of iterations it takes gauss-seidel to converge

# Define the matrix and vector
A = [[3, 1, 1], [1, 4, 1], [2, 3, 7]]
b = [1, 3, 0]

# initial guess, tolerance, and maximum iterations
initial_guess = [0, 0, 0]
tolerance = 1e-6
max_iterations = 50

#  Gauss-Seidel method
for k in range(max_iterations):
    x1 = [0, 0, 0]
    for i in range(3):
        x1[i] = (b[i] - sum([A[i][j] * x1[j] for j in range(i)]) 
                 - sum([A[i][j] * initial_guess[j] for j in range(i+1, 3)])) / A[i][i]
    if all([abs(x1[i] - initial_guess[i]) < tolerance for i in range(3)]):
        break
    initial_guess = x1
k=k+1

# number of iterations
print(k,"\n")
#---------------------------------------------------------------------------------------------------
#2. The number of iterations it takes jacobi method to converge

# Define the matrix A and vector b
A = [[3, 1, 1], [1, 4, 1], [2, 3, 7]]
b = [1, 3, 0]

# Set the initial guess, tolerance, and maximum iterations
initial_guess = [0, 0, 0]
tolerance = 1e-6
max_iterations = 50

# Apply the Jacobi method
for k in range(max_iterations):
    x1 = [0, 0, 0]
    for i in range(3):
        x1[i] = (b[i] - sum([A[i][j] * initial_guess[j] for j in range(3) if j != i])) / A[i][i]
    if all([abs(x1[i] - initial_guess[i]) < tolerance for i in range(3)]):
        break
    initial_guess = x1
k=k+2

# Print the number of iterations
print(k,"\n")
#---------------------------------------------------------------------------------------------------

#3 . Determine the number of iterations necessary to solve f(x) = x3 - x2 + 2 = 0 using newton-raphson from the left side
#Notes from class

def custom_derivative(value):
   return (3 * value * value) - (2 * value)

def newton_raphson(initial_approximation: float, tolerance: float, sequence: str):
    # remember this is an iteration based approach...
    iteration_counter = 0
  
    # finds f
    x = initial_approximation
    f = eval(sequence)
  
    # finds f'
    f_prime = custom_derivative(initial_approximation)

    approximation: float = f / f_prime
    while (abs(approximation) >= tolerance):
        # finds f
        x = initial_approximation
        f = eval(sequence)
        # finds f'
        f_prime = custom_derivative(initial_approximation)
        # division operation
        approximation = f / f_prime
        # subtraction property
        initial_approximation -= approximation
        iteration_counter += 1
      
    return (iteration_counter)
if __name__ == "__main__":
# newton_raphson method
  initial_approximation: float = .5
  tolerance: float = .000001
  sequence: str = "x**3 - (x**2) + 2"

  result =newton_raphson(initial_approximation, tolerance, sequence)
print(result,"\n")
#---------------------------------------------------------------------------------------------------

#4.


def apply_div_dif(matrix):
  size = len(matrix)
  for i in range(2, size):
    for j in range(2, i+2):
    # skip if value is prefilled (we dont want to accidentally recalculate...)
      if j >= len(matrix[i]) or matrix[i][j] != 0:
        continue
      # get left cell entry
      left = matrix[i][j - 1]
      # get diagonal left entry
      diagonal_left =  matrix[i - 1][j - 1]
      # order of numerator is SPECIFIC.
      numerator = left - diagonal_left
    # denominator is current i's x_val minus the starting i's x_val....
      denominator = matrix[i][0] - matrix[i - j + 1][0]
  # something save into matrix
      operation = numerator / denominator
      matrix[i][j] = operation
  return matrix


def hermite_interpolation():
  x_points = [0.0, 1.0, 2.0]
  y_points = [1.0, 2.0, 4.0]
  slopes = [1.06, 1.23, 1.55]
  
  # matrix size changes because of "doubling" up info for hermite
  num_of_points = len(x_points)
  matrix = np.zeros((num_of_points * 2, num_of_points * 2))
  index = 0
  
  for x in range(0, num_of_points *2, 2):
      matrix[x][0] = x_points[index]
      matrix[x + 1][0] = x_points[index]
      index += 1

    # prepopulate y values
  index = 0
  for y in range(0, num_of_points *2, 2):
     matrix[y][1] = y_points[index]
     matrix[y + 1][1] = y_points[index]
     index += 1
    
  index = 0
  for i in range(1, num_of_points *2, 2):
    matrix[i][2] = slopes[index]
    index += 1
    
  filled_matrix = apply_div_dif(matrix)
  print(filled_matrix)
hermite_interpolation()
print("\n")
#---------------------------------------------------------------------------------------------------

#5. The final value of the modified eulers method


# Define the function f(x,y)
def f(x, y):
    return y - x**3 
# set initial conditions
x = 0
y = 0.5

# step size
h = 3/100

# iterate 100 times
for i in range(100):
    # compute yn+1 using modified Euler's method formula
    y = y + (h/2) * (f(x, y) + f(x+h, y + h*f(x, y)))
    # update x
    x = x + h

# final value of y
y_final = y

# print the final value of y
#print( y_final)
print("{:.5f}".format(y_final))



  




