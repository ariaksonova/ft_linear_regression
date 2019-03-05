import pickle
from info import Info
import sys

def main():
	try:
		with open("/tmp/ft_linear_regression", "rb") as file:
			inf = pickle.load(file)
	except Exception:
		print("First step it's run train program!")
		sys.exit(1)
	print("Enter millage:")
	x = input()
	try:
		x = float(x)
		result = inf.wt0 + ((x - inf.xmin) / (inf.xmax - inf.xmin)) * inf.wt1
		print("Price =", result, "and deviation =", inf.errors)
	except Exception:
		print("Not valid input data! Try again with int.")

if __name__ == "__main__":
	main()