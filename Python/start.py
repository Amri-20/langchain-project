def factorial(n):
    if n==0:return  1
    else: return n*factorial(n-1)

n=int(input("Enter the number to calculate factorial: "))
print("The factorial of", n ," is ",factorial(n))